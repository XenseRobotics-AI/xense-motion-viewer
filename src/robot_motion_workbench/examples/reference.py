"""Portable demo and external URDF example using the same public interfaces."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..adapters.reference import ReferenceRobotAdapter, description_from_urdf
from ..control import JointTarget, RobotSession
from ..policy import JointPreview, PolicyChunk, PolicyInput, SampleTime, StreamEvent
from ..urdf import xyz_rpy


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--urdf', type=Path, help='External URDF; omit for the packaged original seven-axis demo')
    parser.add_argument('--tool-link', help='URDF parent link for the declared tool; defaults to final descendant')
    parser.add_argument('--instance-id', default='reference')
    parser.add_argument('--package-root', action='append', default=[], metavar='NAME=DIR')
    parser.add_argument('--base', type=float, nargs=6, default=[.3, -.2, .12, .1, -.15, .35],
                        metavar=('X', 'Y', 'Z', 'R', 'P', 'YAW'), help='world_from_root XYZ metres / RPY radians')
    parser.add_argument('--tool', type=float, nargs=6, default=[.04, -.02, .12, .2, -.1, .3],
                        metavar=('X', 'Y', 'Z', 'R', 'P', 'YAW'), help='link_from_tool XYZ metres / RPY radians')
    parser.add_argument('--save', type=Path, help='Optional real Rerun recording; needs [rerun] extra')
    return parser


def adapter_from_args(args):
    roots = {}
    for value in args.package_root:
        if '=' not in value:
            raise ValueError('--package-root needs NAME=DIR')
        key, path = value.split('=', 1)
        if not key or key in roots:
            raise ValueError('Package names must be nonempty and unique')
        roots[key] = Path(path).resolve(strict=True)
    desc = description_from_urdf(args.urdf, instance_id=args.instance_id,
        tool_link=args.tool_link, package_roots=roots,
        base_transform=xyz_rpy(args.base[:3], args.base[3:]),
        link_from_tool=xyz_rpy(args.tool[:3], args.tool[3:]))
    return ReferenceRobotAdapter(desc)


def goal_positions(adapter):
    """Small static reference inside URDF bounds, with every axis participating."""
    observed = adapter.snapshot().joint_positions
    result = {}
    for i, (n, spec) in enumerate(adapter.capabilities().joints.items()):
        delta = (0.025 if spec.unit == 'm' else 0.28) * (1 if i % 2 == 0 else -1)
        value = observed[n] + delta
        if spec.lower is not None and value < spec.lower:
            value = observed[n] + abs(delta)
        if spec.upper is not None and value > spec.upper:
            value = observed[n] - abs(delta)
        result[n] = float(np.clip(value, spec.lower if spec.lower is not None else -np.inf,
                                 spec.upper if spec.upper is not None else np.inf))
    return result


def axis_effects(adapter):
    """Numerical one-axis-at-a-time FK check; it does not command movement."""
    state = adapter.snapshot()
    tool = adapter.describe().tool_frames['tool']
    original = adapter.model.link_transforms(state.joint_positions)[tool.parent_link]
    result = []
    goals = goal_positions(adapter)
    for n in adapter.describe().joint_names:
        q = dict(state.joint_positions)
        q[n] = goals[n]
        pose = adapter.model.link_transforms(q)[tool.parent_link]
        angle = np.arccos(np.clip((np.trace(original[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1))
        original_tool, changed_tool = original @ tool.link_from_tool, pose @ tool.link_from_tool
        result.append(dict(joint=n, delta_si=q[n] - state.joint_positions[n],
            flange_rotation_rad=float(angle),
            flange_translation_m=float(np.linalg.norm(pose[:3, 3] - original[:3, 3])),
            tool_translation_m=float(np.linalg.norm(changed_tool[:3, 3] - original_tool[:3, 3]))))
    return result


def run_reference(adapter, *, steps=100, save=None):
    """Explicit host-driven reference selection and numerical follower steps."""
    session = RobotSession(adapter)
    initial = session.snapshot()
    effects = axis_effects(adapter)
    names = session.description.joint_names
    start = np.array([initial.joint_positions[n] for n in names])
    goal = goal_positions(adapter)
    end = np.array([goal[n] for n in names])
    offsets = np.linspace(.1, 2., 12)
    fractions = np.linspace(0.08, 1., len(offsets))
    values = start + fractions[:, None] * (end - start)
    chunk = PolicyChunk('reference-1', 'arm', names, offsets, values,
                        SampleTime(0.), 5., 'absolute_joint_positions', 'joint')
    worker, recording = None, None
    publish = lambda event: None
    if save is not None:
        import rerun as rr
        from ..policy_renderer import PolicyRerunRenderer
        from ..streams import VisualizationWorker
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.set_strict_mode(True)
        recording = rr.RecordingStream('rmw-reference')
        recording.save(save)
        renderer = PolicyRerunRenderer(recording, session.description)
        # Finite offline run: enough capacity for every event even if rendering
        # starts after the numerical loop. Interactive applications choose their
        # own bounded capacity and must monitor display-drop counters.
        worker = VisualizationWorker(lambda event: renderer.consume(event, stats=worker.stats),
                                     capacity=steps + 64, close_callback=renderer.close)
        publish = worker.publish
    policy = PolicyInput(session.activate('policy'), publish)
    submissions = []
    ref = None
    closed = True
    try:
        publish(StreamEvent('feedback', SampleTime(0., 0., 'reference_simulation'), 'simulated', state=initial,
                            reason='Synthetic rate-limited follower + display FK, not hardware or dynamics'))
        policy.accept(chunk, preview=JointPreview(names, values, 'Reference example: full joint references + display FK'))
        last_index = None
        for step in range(steps):
            now = adapter.time
            index = min(int(step * len(values) / steps), len(values) - 1)
            if index != last_index:
                policy.select(index, now=now)
                submission = policy.execute_selected(now=now)
                ref = submission.ref
                submissions.append(dict(index=index, sequence=submission.receipt.sequence, host_time=now))
                last_index = index
            adapter.advance(2. / steps)
            state = session.snapshot()
            publish(StreamEvent('feedback', SampleTime(state.time, state.time, 'reference_simulation'),
                'simulated', ref, state=state, reason='Synthetic follower state; explicit simulated clock'))
    finally:
        policy.close()
        if worker is not None:
            closed = worker.close(timeout=30)
    final = session.snapshot()
    result = dict(urdf=str(session.description.urdf_path), joint_names=names,
        base_transform=session.description.base_transform.tolist(),
        tool_transform=session.description.tool_frames['tool'].link_from_tool.tolist(),
        tool_parent=session.description.tool_frames['tool'].parent_link,
        initial_joints=dict(initial.joint_positions), final_joints=dict(final.joint_positions),
        axis_effects=effects, submissions=submissions, simulation_time=final.time,
        origin='hand-authored reference / simulated numerical follower; no dynamics, IK or collision checks',
        sent='unavailable: no transport', planned='unavailable: no robot planner',
        source_sample_time='reference_simulation only; no hardware source sample', hardware_connections=0,
        display=asdict(worker.stats) if worker else None, display_closed=closed)
    if save is not None:
        save.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    if not closed or (worker and worker.stats.errors):
        raise RuntimeError(f'Display did not close cleanly: {result["display"]}')
    return result


def main(argv=None):
    parser = parser_for('Offline generic URDF reference movement; no hardware, dynamics or IK')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args(argv)
    if args.steps < 12:
        parser.error('--steps must be >=12')
    print(json.dumps(run_reference(adapter_from_args(args), steps=args.steps, save=args.save), indent=2))
    return 0


def qt_main(argv=None):
    parser = parser_for('Metadata-driven Qt for an offline reference adapter; no hardware')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--ticks', type=int, help='Exit after this many explicit 20ms simulation ticks')
    parser.add_argument('--exercise', action='store_true', help='Explicitly submit all axis drafts once for smoke verification')
    args = parser.parse_args(argv)
    if args.ticks is not None and args.ticks < 1:
        parser.error('--ticks must be positive')
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from ..qt import ControlPanel
    app = QApplication.instance() or QApplication([])
    adapter = adapter_from_args(args)
    session = RobotSession(adapter)
    inputs = session.activate('qt')
    panel = ControlPanel(inputs)
    panel.setWindowTitle('Offline numerical reference · joint targets only · no IK or hardware')
    recording, renderer = None, None
    if args.save:
        import rerun as rr
        from ..rerun_renderer import RerunRenderer
        args.save.parent.mkdir(parents=True, exist_ok=True)
        recording = rr.RecordingStream('rmw-reference-qt')
        recording.save(args.save)
        renderer = RerunRenderer(recording, timeline='sandbox_elapsed')
        renderer.add_robot(session.description)
    if args.exercise:
        for n, value in goal_positions(adapter).items():
            panel.joint_editors['arm'][n].setValue(session.capabilities.joints[n].to_display(value))
        panel.joint_buttons['arm'].click()
    count = 0
    def tick():
        nonlocal count
        adapter.advance(.02)
        state = panel.refresh()
        if renderer:
            renderer.update(session.description.instance_id, state)
        count += 1
        if args.ticks is not None and count >= args.ticks:
            app.quit()
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(20)
    if not args.no_show:
        panel.show()
    try:
        app.exec()
    finally:
        timer.stop()
        inputs.close()
        if renderer:
            renderer.flush()
            recording.disconnect()
    print(json.dumps(dict(ticks=count, axes=len(session.description.joint_names),
        submissions=session.last_receipt.sequence if session.last_receipt else 0,
        tcp_controls=list(panel.tcp_editors), origin='simulated numerical reference', hardware_connections=0)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
