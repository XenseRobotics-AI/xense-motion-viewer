"""Offline Elite sandbox bridge; all motion semantics stay in libpyelite.sim."""
import argparse
from dataclasses import dataclass, replace
import json
import os
import sys
import importlib.util
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter

import numpy as np

from ..contracts import AttachedMesh, RobotDescription, RobotState, ToolFrame
from ..control import ControlCapabilities, JointControl, JointTarget, RobotSession, TcpControl, TcpTarget
from ..urdf import xyz_rpy


class EliteStationAdapter:
    """Offline targets/state; the existing StationSimulation owns all planning."""
    def __init__(self, simulation, *, instance_id='elite_station'):
        self.simulation = simulation
        from .elite_gripper import EliteGrippers
        if not hasattr(simulation, '_rmw_grippers'):
            simulation._rmw_grippers = EliteGrippers(simulation)
        self.grippers = simulation._rmw_grippers
        names = tuple(f'{side}_{joint}' for side in simulation.arms for joint in simulation.joint_names)
        # Some source URDFs expand mimic fingers for Pinocchio. Keep exactly the
        # source model's representation; only independent finger joints are commanded.
        import xml.etree.ElementTree as ET
        root = ET.parse(simulation.urdf_path).getroot()
        independent = {j.attrib['name'] for j in root.findall('joint')
                       if j.attrib['type'] != 'fixed' and j.find('mimic') is None}
        defaults = {n: v for n, v in simulation.assembly.viewer_joint_defaults().items()
                    if n in independent and n not in self.grippers.specs}
        meshes = []
        if simulation.collision is not None:
            import trimesh
            for shape in simulation.collision.shapes:
                kind = shape['type']
                if kind == 'box':
                    mesh = trimesh.creation.box(extents=shape['size'])
                elif kind == 'convex':
                    mesh = trimesh.Trimesh(vertices=shape['vertices'], faces=shape['faces'], process=False)
                elif kind == 'cylinder':
                    mesh = trimesh.creation.cylinder(radius=shape['radius'], height=shape['length'], sections=24)
                elif kind == 'capsule':
                    mesh = trimesh.creation.capsule(radius=shape['radius'], height=shape['length'], count=[16, 24])
                    # Center from actual bounds: Trimesh's capsule docstring and
                    # implementation disagree about the origin in 5.1.1.
                    mesh.apply_translation([0, 0, -float(mesh.bounds[:, 2].mean())])
                else:
                    raise ValueError(f'Unsupported Elite collision shape: {kind}')
                rgba = (1, 0.6, 0, 0.25) if shape['group'] == 'environment' else (0.15, 0.65, 0.6, 0.3)
                meshes.append(AttachedMesh(shape['name'], shape['frame'], mesh.vertices, mesh.faces,
                    xyz_rpy(shape['xyz'], shape['rpy']), rgba))
        self._description = RobotDescription(
            instance_id, simulation.urdf_path, names + tuple(self.grippers.specs), joint_defaults=defaults,
            tool_frames={side: ToolFrame(f'{side}_tool0', arm.tool) for side, arm in simulation.arms.items()},
            collision_meshes=tuple(meshes),
        )
        self._capabilities = ControlCapabilities(
            joints={**{joint: JointControl('rad', limits[0], limits[1], 'deg', 180 / np.pi)
                       for joint, limits in zip(names, simulation.joint_limits, strict=True)}, **self.grippers.specs},
            groups={**{side: tuple(f'{side}_{joint}' for joint in simulation.joint_names)
                       for side in simulation.arms},
                    **{side + '_gripper': (joint,) for side, joint in self.grippers.drivers.items()}},
            tcp_targets={side: TcpControl(side) for side in simulation.arms},
        )

    def describe(self):
        return self._description

    def capabilities(self):
        return self._capabilities

    def submit(self, command):
        command = self._capabilities.validate(command)
        if isinstance(command, JointTarget):
            if command.group.endswith('_gripper'):
                self.grippers.targets.update(command.positions)
            else:
                self.simulation.set_joint_target(command.group, list(command.positions.values()))
        else:
            self.simulation.set_target(command.tcp, command.world_from_tcp)

    def step(self):
        """One native simulation step; not part of the generic session protocol."""
        if self.simulation.collision_error:
            return self.simulation.step()  # Preserve the native latched error before moving fingers.
        self.grippers.advance()
        return self.simulation.step()

    def reset(self, q=None, side='both'):
        sim = self.simulation
        if q is None:
            sim.reset_to_home()
            self.grippers.reset(home=True)
        else:
            candidate = sim.q.copy()
            for i, arm in enumerate(sim.arms):
                if side in ('both', arm):
                    candidate[i * 6:(i + 1) * 6] = np.asarray(q)[i * 6:(i + 1) * 6]
            if sim.collision is not None and not sim.collision.check(self.grippers.full_q(arm_q=candidate)).safe:
                raise ValueError('Reset rejected with current gripper configuration')
            sim.reset(q, side)
            self.grippers.reset()

    def snapshot(self):
        sim = self.simulation
        positions = {f'{side}_{joint}': value
                     for side, arm in sim.arms.items()
                     for joint, value in zip(sim.joint_names, arm.q, strict=True)}
        positions.update(self.grippers.positions)
        return RobotState(max(arm.time for arm in sim.arms.values()), positions,
                          tcp_poses=sim.tcp_poses(), targets=sim.targets())


@dataclass(frozen=True)
class EliteDisplayStats:
    accepted: int
    rendered: int
    coalesced: int
    pending: int
    errors: int
    last_error: str | None
    closed: bool
    alive: bool
    latest_time: float
    rendered_time: float
    last_lag_s: float
    max_lag_s: float


class _LatestDisplay:
    """One in-flight frame and one replaceable pending frame, capped at 30 Hz.

    Only immutable snapshots/visibility cross this boundary. No SDK, simulation
    or input object is passed to the consumer. A stuck sink cannot block Qt;
    close has a finite join and reports a still-running daemon explicitly.
    """
    def __init__(self, consume, finish, *, hz=30, open_callback=None, follow_callback=None):
        self.consume, self.finish = consume, finish
        self.open_callback = open_callback
        self.open_requested = False
        self.follow_callback = follow_callback
        self.follow_requested = False
        self.period = 1 / hz
        self.condition = Condition()
        self.pending = None
        self.closed = False
        self.accepted = self.rendered = self.coalesced = self.errors = 0
        self.last_error = None
        self.latest_time = self.rendered_time = 0.
        self.last_lag = self.max_lag = 0.
        self.thread = Thread(target=self._run, name='elite-rerun', daemon=True)
        self.thread.start()

    def publish(self, state, visible, *, open_viewer=False, follow_live=False):
        with self.condition:
            if self.closed:
                return False
            self.coalesced += self.pending is not None
            self.pending = (state, visible, perf_counter())
            self.open_requested |= open_viewer
            self.follow_requested |= follow_live
            self.latest_time = state.time
            self.accepted += 1
            self.condition.notify()
            return True

    @property
    def stats(self):
        with self.condition:
            return EliteDisplayStats(self.accepted, self.rendered, self.coalesced, int(self.pending is not None),
                self.errors, self.last_error, self.closed, self.thread.is_alive(), self.latest_time,
                self.rendered_time, self.last_lag, self.max_lag)

    def _run(self):
        next_frame = 0.
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closed)
                    if self.pending is None:
                        break
                    # Wait before taking the mailbox, so newer states replace
                    # the pending one throughout the cadence/slow-sink delay.
                    wait = next_frame - perf_counter()
                    if wait > 0 and not self.closed:
                        self.condition.wait(timeout=wait)
                        if perf_counter() < next_frame and not self.closed:
                            continue
                    state, visible, received = self.pending
                    open_viewer, self.open_requested = self.open_requested, False
                    follow_live, self.follow_requested = self.follow_requested, False
                    self.pending = None
                try:
                    if open_viewer and self.open_callback is not None:
                        self.open_callback(state, visible)
                    self.consume(state, visible)
                    if follow_live and self.follow_callback is not None:
                        self.follow_callback(state, visible)
                    with self.condition:
                        self.rendered += 1
                        self.rendered_time = state.time
                        self.last_lag = perf_counter() - received
                        self.max_lag = max(self.max_lag, self.last_lag)
                except Exception as exc:
                    with self.condition:
                        self.errors += 1
                        self.last_error = f'{type(exc).__name__}: {exc}'
                next_frame = perf_counter() + self.period
        finally:
            try:
                self.finish()
            except Exception as exc:
                with self.condition:
                    self.errors += 1
                    self.last_error = f'{type(exc).__name__}: {exc}'

    def close(self, timeout=1.):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=max(0., timeout))
        return not self.thread.is_alive()


class EliteRerunView:
    """Duck-typed viewer consumed by the existing Elite Qt input window."""
    url = 'Rerun native viewer'

    def __init__(self, simulation, recording, *, instance_id='elite_station', port=9876):
        from ..rerun_renderer import RerunRenderer
        self.adapter = EliteStationAdapter(simulation, instance_id=instance_id)
        self.instance_id = instance_id
        self.renderer = RerunRenderer(recording, timeline='sandbox_elapsed', compact=True)
        self.renderer.add_robot(self.adapter.describe())
        self.port = port
        self._previous_raw = 0.0
        self._time_offset = 0.0
        self._display = None
        self.collision_visible = False
        self._last_state = None
        self._last_follow = 0.
        self.update(simulation)

    def start_display(self):
        if self._display is None:
            self._display = _LatestDisplay(self._consume, self._finish, open_callback=self._open,
                                           follow_callback=self._follow_live)

    @property
    def stats(self):
        return self._display.stats if self._display else EliteDisplayStats(
            self.renderer.update_count, self.renderer.update_count, 0, 0, 0, None, False, False,
            self._last_state.time, self._last_state.time, 0., 0.)

    def _consume(self, state, visible):
        self.renderer.set_collision_visible(visible)
        self.renderer.update(self.instance_id, state)

    def _finish(self):
        try:
            self.renderer.recording.flush(timeout_sec=.5)
        finally:
            try:
                self.renderer.recording.disconnect()
            finally:
                self.renderer.recording._rmw_server = None

    def close(self, timeout=1.):
        if self._display:
            return self._display.close(timeout)
        self._finish()
        return True

    def update(self, simulation):
        if simulation is not self.adapter.simulation:
            raise ValueError('Viewer is bound to a different StationSimulation')
        state = self.adapter.snapshot()
        # A production UI reset restarts planner time. Keep the output clock
        # monotonic across resets, without modifying any planner clock/state.
        if state.time < self._previous_raw:
            self._time_offset += self._previous_raw
        self._previous_raw = state.time
        self._last_state = replace(state, time=state.time + self._time_offset)
        if self._display:
            self._display.publish(self._last_state, self.collision_visible)
        else:
            self._consume(self._last_state, self.collision_visible)

    def set_collision_visible(self, visible):
        self.collision_visible = bool(visible)
        if self._display:
            self._display.publish(self._last_state, self.collision_visible)
        else:
            self.renderer.set_collision_visible(visible)

    def open(self):
        if self._display:
            return self._display.publish(self._last_state, self.collision_visible, open_viewer=True)
        return self._open(self._last_state, self.collision_visible)

    def follow_live(self):
        """A deliberate Qt edit requests current display time, off the UI thread."""
        if getattr(self.renderer.recording, '_rmw_server', None) is None:
            return False
        if self._display:
            return self._display.publish(self._last_state, self.collision_visible, follow_live=True)
        return self._follow_live(self._last_state, self.collision_visible)

    def _follow_live(self, state, visible):
        from rerun.experimental import ViewerClient
        now = perf_counter()
        if now - self._last_follow < .5:
            return False  # Coalesce a continuous mouse drag, not every display tick.
        self._last_follow = now
        recording = self.renderer.recording
        try:
            client = ViewerClient.connect(f'rerun+http://127.0.0.1:{self.port}/proxy')
            active = client.viewer_state().active_recording
            if not (active or '').endswith(':' + recording.get_recording_id()):
                self._open(state, visible)
                active = client.viewer_state().active_recording
            if not (active or '').endswith(':' + recording.get_recording_id()):
                return False
            recording.flush(timeout_sec=.5)
            client.set_time(self.renderer.timeline, duration=state.time, play=True, recording=active)
            return True
        except ConnectionError:
            return False  # A closed Viewer never blocks or rejects a robot target.

    def _open(self, state, visible):
        def keyframe():
            # GrpcServerSink 0.38.1 can evict static messages too. Refresh the
            # model and every dynamic placement only for a new Viewer attachment.
            self.renderer.refresh_static(self.instance_id)
            self.renderer._last_poses.clear()
            self.renderer._frames[self.instance_id].clear()
            self.renderer.set_collision_visible(visible, force=True)
            self.renderer.update(self.instance_id, state)
            self.renderer.recording.flush(timeout_sec=.5)
        return _open_recording_viewer(self.renderer.recording, self.port, prepare=keyframe)


class EliteSandboxRuntime:
    """Offline host scheduling, kept outside the generic frontend/controller.

    Uses the production sandbox's bounded elapsed-time catch-up policy, calling
    StationSimulation.step unchanged. This is not a hardware servo loop.
    """
    def __init__(self, adapter, view=None):
        self.adapter, self.view = adapter, view
        self._clock = None
        self._backlog = 0.0
        self.samples = {}

    def advance(self, steps=1):
        for _ in range(steps):
            self.samples = self.adapter.step()
        if self.view:
            self.view.update(self.adapter.simulation)
        return self.samples

    def poll(self):
        import time
        from libpyelite.sim import _MAX_STEPS_PER_FRAME
        now = time.perf_counter()
        if self._clock is None:
            self._clock = now
            return self.samples
        elapsed, self._clock = now - self._clock, now
        if elapsed > 0.5:
            return self.samples
        period = next(iter(self.adapter.simulation.arms.values())).config.period
        self._backlog = min(self._backlog + elapsed, _MAX_STEPS_PER_FRAME * period)
        steps = int(self._backlog / period)
        self._backlog -= steps * period
        return self.advance(steps) if steps else self.samples

    def reset_to_home(self):
        self.adapter.reset()
        self._clock, self._backlog, self.samples = None, 0.0, {}
        if self.view:
            self.view.update(self.adapter.simulation)


def make_generic_qt_window(simulation, view, *, xyz_range=0.3):
    """Elite host around the public, metadata-driven ControlPanel."""
    _qt_environment()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
    from ..qt import ControlPanel
    app = QApplication.instance() or QApplication([])
    adapter = view.adapter if view else EliteStationAdapter(simulation)
    if view:
        view.start_display()
    session = RobotSession(adapter)
    inputs = session.activate('qt')
    runtime = EliteSandboxRuntime(adapter, view)
    window = QMainWindow()
    window.setWindowTitle('Elite CS66 FK / IK — Rerun sandbox')
    window.resize(1050, 900)
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel('Offline Elite simulation · production native planner · no hardware connection'))
    layout.addWidget(QLabel('Collision guard: ' + ('enabled' if simulation.collision else 'disabled')))
    panel = ControlPanel(inputs, xyz_range=xyz_range)
    layout.addWidget(panel)
    from .elite_gripper import make_gripper_panel
    gripper_panel = make_gripper_panel(inputs, adapter.grippers,
        {side: panel.joint_group_boxes[side].layout() for side in simulation.arms},
        ik_layouts={side: panel.tcp_group_boxes[side].layout() for side in simulation.arms},
        on_submit=view.follow_live if view else None)
    for side in simulation.arms:
        panel.joint_group_boxes[side + '_gripper'].hide()
    window.gripper_panel = gripper_panel
    window.ik_gripper_panel = gripper_panel.ik
    status = QLabel('Observed state comes from simulation; submission does not mean sent or achieved.')
    status.setWordWrap(True)
    layout.addWidget(status)
    open_button = QPushButton('Open Rerun')
    open_button.setEnabled(view is not None and getattr(view.renderer.recording, '_rmw_server', None) is not None)
    if view:
        open_button.clicked.connect(view.open)
    layout.addWidget(open_button)
    collision = QPushButton('Show collision envelopes')
    collision.setCheckable(True)
    collision.setEnabled(view is not None and simulation.collision is not None)
    if view:
        collision.toggled.connect(view.set_collision_visible)
    layout.addWidget(collision)
    home = QPushButton('Reset to home · offline simulation reset')
    layout.addWidget(home)
    window.setCentralWidget(central)
    timer = QTimer(window)
    timer.setInterval(16)

    def display(samples):
        panel.refresh()
        gripper_panel.refresh()
        status.setText('Simulation observations: ' + json.dumps(samples, default=float))

    def advance(steps=4):
        try:
            display(runtime.advance(steps))
        except Exception as exc:
            timer.stop()
            status.setText(f'Simulation paused: {exc}; explicit safe reset required.')
            raise

    def tick():
        try:
            display(runtime.poll())
        except Exception as exc:
            timer.stop()
            status.setText(f'Simulation paused: {exc}; explicit safe reset required.')

    def reset():
        try:
            if not inputs.active:
                raise RuntimeError('Qt no longer owns input; simulation reset rejected')
            runtime.reset_to_home()
            panel.sync_targets()
            panel.refresh()
            gripper_panel.sync()
            status.setText('Offline simulation reset to checked home configuration.')
            timer.start()
        except Exception as exc:
            status.setText(f'Reset rejected: {exc}')

    home.clicked.connect(reset)
    timer.timeout.connect(tick)
    window.panel, window.session, window.runtime = panel, session, runtime
    window.render_timer, window.advance, window._reset_home = timer, advance, reset
    window.collision_toggle = collision
    # Stop host work when the window is closed, even if the caller retains it.
    def close(event):
        timer.stop()
        inputs.close()
        if view and not view.close(timeout=1.):
            status.setText('Rerun close timed out; display thread is still draining (control stopped).')
        event.accept()
    window.closeEvent = close
    timer.start()
    return app, window


def make_qt_window(simulation, view, *, xyz_range=0.3):
    """Keep the production Elite sliders; route their targets via RobotSession.

    This deliberately Elite-specific UI is optional SDK code, outside the
    generic frontend. The facade owns named mapping; no IK is introduced here.
    """
    _qt_environment()
    from libpyelite.sim import _load_qt, make_window
    qt = _load_qt()
    app = qt['QApplication'].instance() or qt['QApplication']([])
    adapter = view.adapter if view else EliteStationAdapter(simulation)
    if view:
        view.start_display()
    session = RobotSession(adapter)
    inputs = session.activate('qt')
    runtime = EliteSandboxRuntime(adapter, view)

    class Facade:
        def __getattr__(self, key):
            return getattr(simulation, key)

        def set_joint_target(self, side, q):
            names = session.capabilities.groups[side]
            return inputs.submit(JointTarget(side, dict(zip(names, q, strict=True))))

        def set_target(self, side, pose):
            return inputs.submit(TcpTarget(side, pose))

        def reset(self, q, side='both'):
            if not inputs.active:
                raise ValueError('Qt input ownership is inactive')
            adapter.reset(q, side)

        def reset_to_home(self):
            if not inputs.active:
                raise ValueError('Qt input ownership is inactive')
            runtime.reset_to_home()

    class ViewerFacade:
        url = 'Rerun native viewer'

        def update(self, model):
            view.update(simulation)

        def set_collision_visible(self, visible):
            view.set_collision_visible(visible)

    window = make_window(qt, Facade(), ViewerFacade() if view else None, xyz_range)
    window.setWindowTitle('Elite CS66 FK / IK — Rerun offline sandbox')
    window.mesh_status.setText('Rerun · offline simulation · no hardware connection')
    for button in window.findChildren(qt['QPushButton']):
        if 'MeshCat' in button.text():
            button.setText('打开 Rerun')
            button.clicked.disconnect()
            if view:
                button.clicked.connect(view.open)
                button.setEnabled(getattr(view.renderer.recording, '_rmw_server', None) is not None)
    receipt_label = qt['QLabel']('Input owner: qt · target submission does not mean sent or achieved.')
    receipt_label.setWordWrap(True)
    window.centralWidget().layout().addWidget(receipt_label)
    from .elite_gripper import make_gripper_panel
    gripper_panel = make_gripper_panel(inputs, adapter.grippers,
        {side: getattr(window, side + '_joints').widget.layout() for side in simulation.arms},
        ik_layouts={side: getattr(window, side + '_pose').widget.layout() for side in simulation.arms},
        on_submit=view.follow_live if view else None)
    window.gripper_panel = gripper_panel
    window.ik_gripper_panel = gripper_panel.ik
    original_sync = window._sync_from_q

    def sync():
        original_sync()
        gripper_panel.sync()
    window._sync_from_q = sync
    original_fk, original_solve = window._apply_fk, window._solve

    def submit(callback, side):
        previous = session.last_receipt
        try:
            callback(side)
        except Exception as exc:
            receipt_label.setText(f'Target rejected: {exc}')
            return
        receipt = session.last_receipt
        if receipt is not previous:
            receipt_label.setText(f'Target submitted #{receipt.sequence} by {receipt.source}; sent/achieved unknown.')
            if isinstance(receipt.command, JointTarget):
                window.fk_status.setText('关节目标已提交；不代表已发送或已到达。')

    window._apply_fk = lambda side: submit(original_fk, side)
    window._solve = lambda side: submit(original_solve, side)

    def display(samples):
        gripper_panel.refresh()
        # Only observations update these controls; set_q blocks widget signals.
        window._syncing = True
        try:
            for side, arm in simulation.arms.items():
                if arm.joint_target is None:
                    getattr(window, side + '_joints').set_q(arm.q)
        finally:
            window._syncing = False
        if any(arm.joint_target is not None for arm in simulation.arms.values()):
            window.fk_status.setText('仿真实际 q [deg] = ' + np.array2string(np.degrees(simulation.q), precision=2))
        results = [(side, sample, window._classify(side, sample)) for side, sample in samples.items()]
        window.ik_status.setText('\n'.join(
            f'{side}: {result[1]}, pos={sample["position_error_m"]:.2e} m, '
            f'ori={sample["orientation_error_rad"]:.2e} rad, QP failure={sample["qp_failed"]}, '
            f'solve={sample["solve_time_s"] * 1000:.3f} ms' +
            (f', collision={sample["collision_distance_m"] * 1000:.2f} mm, '
             f'constraints={sample["collision_constraints"]}' if 'collision_distance_m' in sample else '')
            for side, sample, result in results))
        levels = {result[0] for _, _, result in results}
        color = '#b00020' if 'error' in levels else '#1b7f2a' if levels == {'ok'} else '#a15c00'
        window.ik_status.setStyleSheet(f'font-family: monospace; color: {color};')

    def advance(steps=4):
        try:
            display(runtime.advance(steps))
        except Exception as exc:
            window.render_timer.stop()
            window.ik_status.setText(f'仿真已暂停: {exc}; reset explicitly to a checked safe configuration.')

    def tick():
        try:
            display(runtime.poll())
            if view:
                stats = view.stats
                window.mesh_status.setText(
                    f'Rerun ≤30 Hz · 合并 {stats.coalesced} 帧 · 显示耗时 {stats.last_lag_s*1000:.0f} ms'
                    + (f' · 错误: {stats.last_error}' if stats.errors else ''))
        except Exception as exc:
            window.render_timer.stop()
            window.ik_status.setText(f'仿真已暂停: {exc}; reset explicitly to a checked safe configuration.')

    window.render_timer.timeout.disconnect()
    window.render_timer.timeout.connect(tick)
    window.advance = advance
    window.session, window.runtime, window.receipt_status = session, runtime, receipt_label
    def close(event):
        window.render_timer.stop()
        inputs.close()
        if view and not view.close(timeout=1.):
            receipt_label.setText('Rerun close timed out; display thread is still draining (control stopped).')
        event.accept()
    window.closeEvent = close
    return app, window


def _parser(qt=False):
    parser = argparse.ArgumentParser(description='Offline production Elite simulation with Rerun output')
    parser.add_argument('--station', choices=['diagonal-07', 'diagonal-08'])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--collision-profile', help='auto, a profile JSON, or off; same semantics as elite-ik-sim')
    parser.add_argument('--instance-id', default='elite_station')
    parser.add_argument('--save', type=Path, required=not qt)
    if qt:
        parser.add_argument('--ui', choices=['elite', 'generic'], default='elite')
        parser.add_argument('--no-viewer', action='store_true')
        parser.add_argument('--no-show', action='store_true', help='Exercise public command/Qt callbacks and exit')
        parser.add_argument('--port', type=int, default=9876)
        parser.add_argument('--xyz-range', type=float, default=0.3)
    else:
        parser.add_argument('--show-collision', action='store_true')
        parser.add_argument('--steps', type=int, default=100)
        parser.add_argument('--jog-m', type=float, default=0.0, help='Optional small world-X target delta for one arm')
    return parser


def _simulation(args):
    try:
        from libpyelite.sim import StationSimulation, load_recipe
    except ImportError as exc:
        raise RuntimeError(
            'Elite requires the current-production-source libpyelite wheel and its dependencies. '
            'See README.md development/dependencies: install the matching vendor/wheels/libpyelite wheel, '
            'then python -m pip install "robot-motion-workbench[elite]". '
            f'A same-version older wheel is insufficient. Original import error: {exc}'
        ) from exc
    settings = None if args.recipe is None else load_recipe(args.recipe)
    station = args.station or (settings or {}).get('native_assembly', 'diagonal-08')
    if settings and settings.get('native_assembly', station) != station:
        raise ValueError('--station and recipe native_assembly must agree')
    collision = args.collision_profile
    if collision is None:
        collision = (settings or {}).get('native_collision_profile')
    if collision in ('off', 'none', 'null'):
        collision = None
    return StationSimulation(station, settings=settings, collision_profile=collision)


def _open_recording_viewer(recording, port, *, prepare=None):
    """Display-owner operation; never replaces sinks or reopens the file."""
    import time
    import rerun as rr
    from rerun.experimental import ViewerClient
    server = getattr(recording, '_rmw_server', None)
    if server is None:
        raise RuntimeError('Live viewing was not configured; restart without --no-viewer or replay the saved RRD')
    deadline = perf_counter() + 3.
    while True:
        try:
            rr.spawn(port=port, connect=False, recording=recording,
                     executable_path=str(Path(sys.executable).parent / 'rerun'))
            client = ViewerClient.connect(f'rerun+http://127.0.0.1:{port}/proxy')
            if (client.viewer_state().active_recording or '').endswith(':' + recording.get_recording_id()):
                return False  # Already viewing this recording: preserve camera.
            if prepare is not None:
                prepare()
            client.open_url(server.uri)
            return True
        except ConnectionError:
            # A just-closed Viewer may still own its port while shutting down.
            if perf_counter() >= deadline:
                raise
            time.sleep(.05)


def _recording(path, *, viewer_port=None):
    import rerun as rr
    rr.set_strict_mode(True)
    # SDK default microbatch tick is 200ms, visible as drag latency even when
    # Python logging is fast. Match the Elite display's 30Hz delivery cadence.
    recording = rr.RecordingStream('robot-motion-workbench-elite',
                                   batcher_config=rr.ChunkBatcherConfig(flush_tick=.033))
    sinks = []
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(rr.FileSink(path))
    if viewer_port is not None:
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            server_port = sock.getsockname()[1]
        recording._rmw_server = rr.GrpcServerSink(bind_ip='127.0.0.1', port=server_port,
                                                 server_memory_limit='64MiB')
        sinks.append(recording._rmw_server)
    if sinks:
        recording.set_sinks(*sinks)
    recording._rmw_save_path = path
    if viewer_port is not None:
        _open_recording_viewer(recording, viewer_port)
    return recording


def record_main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.steps < 0 or not np.isfinite(args.jog_m):
        parser.error('--steps must be nonnegative and --jog-m finite')
    sim = _simulation(args)
    recording = _recording(args.save)
    view = EliteRerunView(sim, recording, instance_id=args.instance_id)
    view.set_collision_visible(args.show_collision)
    samples = None
    if args.jog_m:
        side = next(iter(sim.arms))
        target = sim.targets()[side].copy()
        target[0, 3] += args.jog_m
        RobotSession(view.adapter).activate('programmatic').submit(TcpTarget(side, target))
    for _ in range(args.steps):
        samples = view.adapter.step()
        view.update(sim)
    view.renderer.flush()
    print(json.dumps({'recording': str(args.save.resolve()), 'steps': args.steps,
        'dof': view.adapter.describe().dof,
        'geometry_count': view.renderer.geometry_log_count,
        'collision_count': len(view.adapter.describe().collision_meshes),
        'render_updates': view.renderer.update_count,
        'simulation_seconds': view.adapter.snapshot().time,
        'planner_samples': samples or {}}, indent=2))
    return 0


def _qt_environment():
    # Configure plugins from the wheel that supplies Qt. Never call the vendor
    # CLI bootstrap: it execs libpyelite.sim and would launch the Meshcat CLI.
    spec = importlib.util.find_spec('PySide6')
    if spec is None or spec.origin is None:
        raise RuntimeError('Install robot-motion-workbench[qt] for the Qt input UI')
    root = Path(spec.origin).parent / 'Qt'
    if (root / 'plugins/platforms').is_dir():
        os.environ['QT_PLUGIN_PATH'] = str(root / 'plugins')
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(root / 'plugins/platforms')
    return root / 'lib'


def qt_main(argv=None):
    qt_lib = _qt_environment()
    paths = [p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p]
    if (argv is None and sys.platform.startswith('linux') and
            (qt_lib / 'libQt6Core.so.6').is_file() and (not paths or paths[0] != str(qt_lib))):
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = ':'.join([str(qt_lib), *[p for p in paths if p != str(qt_lib)]])
        os.execve(sys.executable, [sys.executable, '-m', 'robot_motion_workbench.adapters.elite', *sys.argv[1:]], env)
    parser = _parser(qt=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.xyz_range) or args.xyz_range <= 0:
        parser.error('--xyz-range must be finite and positive')
    sim = _simulation(args)
    recording = _recording(args.save, viewer_port=None if args.no_viewer else args.port)
    view = EliteRerunView(sim, recording, instance_id=args.instance_id, port=args.port)
    factory = make_qt_window if args.ui == 'elite' else make_generic_qt_window
    app, window = factory(sim, view, xyz_range=args.xyz_range)
    if args.no_show:
        window.render_timer.stop()
        side = next(iter(sim.arms))
        before = sim.q.copy()
        if args.ui == 'generic':
            panel = window.panel
            joint = next(iter(panel.joint_editors[side]))
            editor = panel.joint_editors[side][joint]
            editor.setValue(editor.value() + np.degrees(0.001))
            panel.joint_buttons[side].click()
        else:
            control = getattr(window, side + '_joints').controls[0]
            control.spin.setValue(control.value() + np.degrees(0.001))
        np.testing.assert_array_equal(sim.q, before)  # input is a target, never a teleport
        assert window.session.last_receipt.status == 'submitted'
        np.testing.assert_allclose(sim.arms[side].joint_target, list(window.session.last_receipt.command.positions.values()))
        window.advance()
        if args.ui == 'generic':
            panel.sync_targets()
            editor = panel.tcp_editors[side]['x']
            editor.setValue(editor.value() + 0.001)
            panel.tcp_buttons[side].click()
        else:
            control = getattr(window, side + '_pose').controls['x']
            control.spin.setValue(control.value() + 0.001)
        assert sim.arms[side].joint_target is None
        np.testing.assert_allclose(sim.targets()[side], window.session.last_receipt.command.world_from_tcp, atol=1e-12)
        window.advance()
        for selected in sim.arms:
            value = 250 if selected == 'left' else 750
            window.gripper_panel.sliders[selected].setValue(value)
            assert window.session.last_receipt.command.group == selected + '_gripper'
            assert window.ik_gripper_panel.sliders[selected].value() == value
        window.advance(130)
        assert view.adapter.grippers.opening('left') == .25
        assert view.adapter.grippers.opening('right') == .75
        for selected in sim.arms:
            value = 40 if selected == 'left' else 60
            window.ik_gripper_panel.editors[selected].setValue(value)
            assert window.session.last_receipt.command.group == selected + '_gripper'
            assert window.gripper_panel.sliders[selected].value() == value * 10
        window.advance(130)
        assert view.adapter.grippers.opening('left') == .4
        assert view.adapter.grippers.opening('right') == .6
        toggle = window.collision_toggle if args.ui == 'generic' else next(
            (button for button in window.findChildren(__import__('PySide6.QtWidgets', fromlist=['QPushButton']).QPushButton)
             if button.text() == '显示碰撞包络'), None)
        if toggle is not None:
            toggle.setChecked(True)
        else:
            view.set_collision_visible(True)
        assert view.collision_visible
        if toggle is not None:
            toggle.setChecked(False)
        else:
            view.set_collision_visible(False)
        assert not view.collision_visible
        collision_toggled = True
        window._reset_home()
        window.render_timer.stop()
        np.testing.assert_allclose(sim.q, sim.home_q)
        assert all(view.adapter.grippers.opening(selected) == 1 for selected in sim.arms)
        for gripper_panel in (window.gripper_panel, window.ik_gripper_panel):
            assert all(slider.value() == 1000 for slider in gripper_panel.sliders.values())
        window.advance()
        app.processEvents()
        window.close()
        print(json.dumps({'qt_smoke': 'passed', 'render_updates': view.renderer.update_count,
                          'simulation_seconds': view.adapter.snapshot().time,
                          'joint_input_checked': True, 'cartesian_input_checked': True,
                          'gripper_input_checked': True,
                          'ik_gripper_input_checked': True,
                          'collision_toggle_checked': collision_toggled, 'reset_checked': True}))
        return 0
    window.show()
    try:
        result = app.exec()
    finally:
        closed = view.close(timeout=1.)
        if not closed or view.stats.errors:
            print(f'Rerun display close={closed}; stats={view.stats}', file=sys.stderr)
    return result if closed and not view.stats.errors else 1


if __name__ == '__main__':
    raise SystemExit(qt_main())
