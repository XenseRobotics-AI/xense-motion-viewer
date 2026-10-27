"""Rendering backpressure must not become robot-input backpressure."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, get_ident
import time

import numpy as np
import pytest

pytest.importorskip('libpyelite')
from libpyelite.sim import StationSimulation
from robot_motion_workbench import RobotState
from robot_motion_workbench.adapters.elite import EliteStationAdapter, EliteSandboxRuntime, _LatestDisplay
from robot_motion_workbench.rerun_renderer import RerunRenderer
from robot_motion_workbench.urdf import URDFModel, xyz_rpy

pytestmark = pytest.mark.elite


def test_blocked_display_keeps_only_last_state_and_close_is_bounded():
    entered, release = Event(), Event()
    observed = []
    owner = get_ident()
    def consume(state, visible):
        assert get_ident() != owner
        observed.append((state.time, state.joint_positions['q'], visible))
        if len(observed) == 1:
            entered.set()
            assert release.wait(2)
    worker = _LatestDisplay(consume, lambda: None, hz=1000)
    worker.publish(RobotState(0, {'q': 0}), False)
    assert entered.wait(1)
    start = time.perf_counter()
    for n in range(1, 101):
        assert worker.publish(RobotState(n, {'q': n}), n % 2 == 0)
    assert time.perf_counter() - start < .1
    assert worker.stats.pending == 1 and worker.stats.coalesced == 99
    assert not worker.close(timeout=.01)
    assert worker.stats.alive and worker.stats.closed
    assert not worker.publish(RobotState(101, {'q': 101}), False)
    release.set()
    assert worker.close(timeout=1)
    assert observed == [(0., 0., False), (100., 100., True)]
    assert worker.stats.rendered_time == worker.stats.latest_time == 100.
    assert not worker.stats.alive and worker.stats.errors == 0


def test_display_errors_are_observable_and_finish_runs_once():
    finished = []
    def fail(state, visible):
        raise RuntimeError('sink blocked then failed')
    worker = _LatestDisplay(fail, lambda: finished.append(True))
    worker.publish(RobotState(0, {'q': 0}), False)
    assert worker.close(timeout=1)
    assert worker.close(timeout=1)
    assert worker.stats.errors == 1 and 'sink blocked then failed' in worker.stats.last_error
    assert finished == [True]


@pytest.mark.parametrize('hz', [30, 60])
def test_display_interval_includes_processing_time_and_handles_overruns(monkeypatch, hz):
    from threading import Condition
    from types import SimpleNamespace
    from robot_motion_workbench.adapters import elite

    clock, starts = [0.], []
    period = 1 / hz
    costs = [.4, 1.5, .4, .4]

    class ClockCondition(Condition):
        def wait(self, timeout=None):
            assert timeout is not None
            clock[0] += timeout

    monkeypatch.setattr(elite, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(elite, 'Condition', ClockCondition)
    # Drive the real worker loop with a controlled clock, without wall-time
    # tolerances that can fail merely because the test machine is busy.
    monkeypatch.setattr(elite, 'Thread', lambda **kwargs: SimpleNamespace(
        start=lambda: None, join=lambda timeout: None, is_alive=lambda: False))

    def consume(state, visible):
        starts.append(clock[0])
        clock[0] += costs[len(starts) - 1] * period
        if len(starts) < len(costs):
            worker.publish(RobotState(clock[0], {'q': 0}), False)
        else:
            worker.close()

    worker = _LatestDisplay(consume, lambda: None, hz=hz)
    worker.publish(RobotState(0, {'q': 0}), False)
    worker._run()
    np.testing.assert_allclose(starts, np.array([0, 1, 2.5, 3.5]) * period, atol=1e-12)
    assert worker.stats.errors == 0 and worker.stats.rendered == 4


def test_policy_coalesces_pending_feedback_without_discarding_send_or_origin_change():
    from robot_motion_workbench import SampleTime, StreamEvent, VisualizationWorker
    entered, release = Event(), Event()
    observed = []
    def consume(event):
        observed.append((event.stage, event.timing.received_at, event.origin))
        if len(observed) == 1:
            entered.set(); assert release.wait(2)
    worker = VisualizationWorker(consume, capacity=5, coalesce_feedback=True)
    def feedback(t, origin='simulated'):
        return StreamEvent('feedback', SampleTime(t), origin, state=RobotState(t, {'q': t}))
    worker.publish(feedback(0)); assert entered.wait(1)
    worker.publish(feedback(1))
    worker.publish(StreamEvent('sent', SampleTime(1.5), 'reference', reason='Actual send evidence unavailable'))
    worker.publish(feedback(2))
    worker.publish(feedback(2.5))
    worker.publish(feedback(3, 'measured'))
    worker.publish(feedback(4))
    release.set(); assert worker.close(timeout=1)
    assert observed == [('feedback', 0., 'simulated'), ('feedback', 1., 'simulated'),
                        ('sent', 1.5, 'reference'), ('feedback', 2.5, 'simulated'),
                        ('feedback', 3., 'measured'), ('feedback', 4., 'simulated')]
    assert worker.stats.coalesced == 1 and worker.stats.errors == 0


def test_rendering_preserves_every_native_step_and_collision_sample(tmp_path):
    import rerun as rr
    from robot_motion_workbench.adapters.elite import EliteRerunView
    actual = StationSimulation(collision_profile='auto')
    direct = StationSimulation(collision_profile='auto')
    for sim in (actual, direct):
        q = sim.arms['left'].q.copy(); q[0] += .01
        sim.set_joint_target('left', q)
    rec = rr.RecordingStream('native-unmodified'); rec.save(tmp_path/'native.rrd')
    view = EliteRerunView(actual, rec)
    view.start_display()
    for i in range(120):
        if i == 60:
            target = direct.tcp_poses()['left'].copy(); target[0, 3] += .002
            for sim in (actual, direct):
                sim.set_target('left', target)
        a, b = actual.step(), direct.step()
        view.update(actual)
        np.testing.assert_array_equal(actual.q, direct.q)
        assert actual.last_collision.safe
        for side in a:
            assert {k: v for k, v in a[side].items() if k != 'solve_time_s'} == {
                k: v for k, v in b[side].items() if k != 'solve_time_s'}
    assert view.close(timeout=2)
    assert view.stats.rendered_time == view.stats.latest_time


def test_compact_recording_preserves_fk_base_mimics_tools_and_exact_collision_batches(tmp_path):
    import rerun as rr
    from rerun.chunk import RrdReader
    sim = StationSimulation(collision_profile='auto')
    adapter = EliteStationAdapter(sim)
    description = replace(adapter.describe(), base_transform=xyz_rpy((.2, -.1, .3), (.1, .2, .3)))
    model = URDFModel(description)
    path = tmp_path/'compact.rrd'
    recording = rr.RecordingStream('compact-correctness'); recording.save(path)
    renderer = RerunRenderer(recording, compact=True)
    renderer.add_robot(description)
    state = adapter.snapshot()
    renderer.update(description.instance_id, state)
    moved = {n: q + .01 for n, q in state.joint_positions.items()}
    override = xyz_rpy((-.1, .2, .4), (-.2, .1, .4))
    renderer.update(description.instance_id, RobotState(1, moved, override))
    renderer.set_collision_visible(True)
    view_id = renderer._spatial_view.id
    renderer.set_collision_visible(False)
    assert renderer._spatial_view.id == view_id
    recording.flush(); recording.disconnect()
    placements = model.link_transforms(moved, override)
    local, parents = {}, {}
    for chunk in RrdReader(path).stream():
        batch = chunk.to_record_batch()
        if 'Transform3D:mat3x3' not in batch.schema.names:
            continue
        for rotation, translation, parent in zip(batch.column('Transform3D:mat3x3').to_pylist(),
                batch.column('Transform3D:translation').to_pylist(), batch.column('Transform3D:parent_frame').to_pylist(), strict=True):
            pose = np.eye(4); pose[:3, :3] = np.array(rotation[0]).reshape(3, 3, order='F'); pose[:3, 3] = translation[0]
            local[chunk.entity_path] = pose
            parents[chunk.entity_path] = '/' + parent[0].removeprefix('tf#/')
    def world(path):
        return np.eye(4) if path == '/world' else world(parents[path]) @ local[path]
    for n, expected in placements.items():
        np.testing.assert_allclose(world('/' + renderer.link_path(description.instance_id, n)), expected, atol=2e-7)
    for key, tool in description.tool_frames.items():
        path = '/' + renderer.instance_path(description.instance_id) + '/tools/' + key
        np.testing.assert_allclose(world(path), placements[tool.parent_link] @ tool.link_from_tool, atol=2e-7)
    source = {m.name: m for m in description.collision_meshes}
    batches = [m for layer, m in renderer._geometries[description.instance_id] if layer == 'collision']
    assert len(source) == 351 and len(batches) == 24
    assert sum(len(m.triangles) for m in batches) == sum(len(m.triangles) for m in source.values()) == 8148
    seen = []
    for batch in batches:
        members = renderer.collision_batches[description.instance_id + '/' + batch.name]
        expected_vertices, expected_faces, offset = [], [], 0
        for name in members:
            mesh = source[name]; seen.append(name)
            assert mesh.parent_link == batch.parent_link and mesh.rgba == batch.rgba
            expected_vertices.append(mesh.vertices @ mesh.link_from_mesh[:3, :3].T + mesh.link_from_mesh[:3, 3])
            expected_faces.append(mesh.triangles + offset); offset += len(mesh.vertices)
        np.testing.assert_allclose(batch.vertices, np.concatenate(expected_vertices), atol=1e-14)
        np.testing.assert_array_equal(batch.triangles, np.concatenate(expected_faces))
    assert sorted(seen) == sorted(source)


def test_real_qt_remains_responsive_with_blocked_rerun(tmp_path):
    spec = importlib.util.find_spec('PySide6')
    if spec is None:
        pytest.skip('Qt extra missing')
    qt = Path(spec.origin).parent/'Qt'
    env = os.environ | {'QT_QPA_PLATFORM': 'offscreen', 'QT_PLUGIN_PATH': str(qt/'plugins'),
        'QT_QPA_PLATFORM_PLUGIN_PATH': str(qt/'plugins/platforms'), 'LD_LIBRARY_PATH': str(qt/'lib')}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('elite_responsive_probe.py')), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
