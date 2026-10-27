import json
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion_workbench.adapters.elite_trajectory import (
    DEFAULT_TRAJECTORY, RecordedTrajectory, tcp_poses_from_actions,
)


FEATURES = [f'{side}_tcp.{key}' for side in ('left', 'right')
            for key in ('x', 'y', 'z', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6')]


def export_fixture(path, *, timestamps=(10., 10.03, 10.10), **metadata):
    meta = dict(format_version=1, robot_type='bi_elite_cs66_rt', frame='world',
                position_unit='m', rotation_encoding='first_two_columns', source_column='action',
                fps=30, frame_count=3, episode_index=7, feature_names=FEATURES)
    meta.update(metadata)
    # The arms have distinct translations and a +90 degree Z rotation.
    actions = np.tile([1, 2, 3, 0, 1, 0, -1, 0, 0, 4, 5, 6, 0, 1, 0, -1, 0, 0], (3, 1))
    np.savez(path, metadata=np.array(json.dumps(meta)), timestamps=timestamps,
             frame_indices=[300, 301, 302], actions=actions)
    return path


def test_column_rotation_convention_and_named_mapping():
    names = ['left_gripper.pos', *reversed(FEATURES)]
    values = dict(zip(FEATURES, [1, 2, 3, 0, 2, 0, -3, .01, 0,
                                4, 5, 6, 1, 0, 0, 0, 1, 0], strict=True))
    values['left_gripper.pos'] = .8
    actions = [[values[name] for name in names]]
    left = tcp_poses_from_actions(actions, names, 'left')[0]
    np.testing.assert_allclose(left, [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    right = tcp_poses_from_actions(actions, names, 'right')[0]
    np.testing.assert_array_equal(right[:3, 3], [4, 5, 6])
    np.testing.assert_array_equal(right[:3, :3], np.eye(3))


def test_source_timestamps_and_final_short_chunk(tmp_path):
    trajectory = RecordedTrajectory(export_fixture(tmp_path / 'sample.npz'))
    caps = SimpleNamespace(groups={'left': tuple(f'j{i}' for i in range(6))})
    assert trajectory.index_at(.02) == 0
    assert trajectory.index_at(.04) == 1
    assert trajectory.index_at(.09) == 1  # Original irregular timestamps, not a fabricated 30 Hz grid.
    assert trajectory.index_at(2.) == 2
    chunk, end = trajectory.chunk(caps, 1, now=.04, size=60)
    assert end == 3
    assert chunk.timing.source_time == 10.03
    assert chunk.timing.source_clock == 'dataset_episode_7_seconds'
    np.testing.assert_allclose(chunk.time_offsets, [0, .07])
    assert chunk.origin == 'reference' and chunk.frame == 'world'
    np.testing.assert_array_equal(chunk.target(0).world_from_tcp, trajectory.poses[1])
    assert 'frame-301' in chunk.chunk_id
    with pytest.raises(ValueError):
        trajectory.poses[0, 0, 0] = 1


@pytest.mark.parametrize('changes', [
    {'timestamps': [0, 0, 1]}, {'timestamps': [0, 1, np.nan]},
    {'frame': 'base'}, {'position_unit': 'mm'}, {'rotation_encoding': 'rows'},
    {'source_column': 'observation.state'}, {'frame_count': 4}, {'fps': 0},
])
def test_bad_recording_is_rejected_before_control(tmp_path, changes):
    path = export_fixture(tmp_path / 'invalid.npz', **changes)
    with pytest.raises(ValueError):
        RecordedTrajectory(path)


@pytest.mark.parametrize('rotation', [[0, 0, 0, 0, 1, 0], [1, 0, 0, 2, 0, 0]])
def test_degenerate_rotations_are_rejected(rotation):
    values = [[0, 0, 0, *rotation, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
    with pytest.raises(ValueError, match='Degenerate'):
        tcp_poses_from_actions(values, FEATURES, 'left')


def test_bundled_episode_preserves_source_values_and_needs_no_mount():
    trajectory = RecordedTrajectory()
    assert len(trajectory.times) == 4451 and trajectory.fps == 30
    assert trajectory.metadata['episode_index'] == 0
    assert trajectory.duration == pytest.approx(148.36666, abs=1e-5)
    assert len(trajectory.metadata['source_data_sha256']) == 64
    with np.load(DEFAULT_TRAJECTORY, allow_pickle=False) as archive:
        np.testing.assert_array_equal(trajectory.poses[:, :3, 3], archive['actions'][:, :3])
        right = RecordedTrajectory(arm='right')
        np.testing.assert_array_equal(right.poses[:, :3, 3], archive['actions'][:, 10:13])
        np.testing.assert_array_equal(trajectory.gripper_openings, archive['actions'][:, 9])
        np.testing.assert_array_equal(right.gripper_openings, archive['actions'][:, 19])
    np.testing.assert_allclose(np.linalg.det(trajectory.poses[:, :3, :3]), 1, atol=1e-12)


@pytest.mark.elite
@pytest.mark.parametrize(('hz', 'period'), [(30, .004), (60, .008)])
def test_display_samples_follow_time_grid_and_include_final_state(monkeypatch, capsys, hz, period):
    pytest.importorskip('libpyelite')
    pytest.importorskip('rerun')
    from libpyelite.sim import StationSimulation
    from robot_motion_workbench.adapters import elite
    from robot_motion_workbench.adapters.elite_policy import main
    from robot_motion_workbench.streams import VisualizationWorker

    settings = dict(StationSimulation().settings, servoj_time=period)
    simulation = StationSimulation(settings=settings, collision_profile='auto')
    native_period = simulation.arms['left'].config.period
    monkeypatch.setattr(elite, '_simulation', lambda args: simulation)
    feedback = []
    publish = VisualizationWorker.publish

    def capture(worker, event):
        if event.stage == 'feedback':
            feedback.append(event)
        return publish(worker, event)

    monkeypatch.setattr(VisualizationWorker, 'publish', capture)
    assert main(['--no-viewer', '--fast', '--steps', '127', '--display-hz', str(hz)]) == 0
    result = json.loads(capsys.readouterr().out)
    end = result['simulation_time']
    # Every requested time maps to the first native state at or after it.
    # The last step is sampled even when it lies between regular deadlines.
    grid = np.arange(1, int(end * hz) + 1) / hz
    expected = np.r_[0., np.ceil(grid / native_period) * native_period, end]
    np.testing.assert_allclose([e.state.time for e in feedback], expected, atol=1e-9)
    assert result['display_hz'] == hz
    assert dict(feedback[-1].state.joint_positions) == result['final_joints']


@pytest.mark.elite
@pytest.mark.parametrize('arm', ['both', 'left', 'right'])
def test_cli_submits_recorded_actions_at_source_times_across_chunk_boundary(tmp_path, capsys, monkeypatch, arm):
    pytest.importorskip('libpyelite')
    pytest.importorskip('rerun')
    from robot_motion_workbench.adapters.elite_policy import main
    from robot_motion_workbench.adapters.elite import EliteStationAdapter
    arms = ('left', 'right') if arm == 'both' else (arm,)
    trajectories = {side: RecordedTrajectory(arm=side) for side in arms}
    original_step = EliteStationAdapter.step
    steps_checked = []
    def step(adapter):
        now = adapter.snapshot().time
        for side in arms:
            trajectory = trajectories[side]
            index = trajectory.index_at(now)
            # Both native targets must already contain the same source row
            # before the station's collision coordinator advances either arm.
            np.testing.assert_allclose(adapter.simulation.targets()[side], trajectory.poses[index], atol=1e-12)
            gripper = adapter.grippers.target(side, trajectory.gripper_openings[index])
            for joint, target in gripper.positions.items():
                assert adapter.grippers.targets[joint] == target
        steps_checked.append(now)
        return original_step(adapter)
    monkeypatch.setattr(EliteStationAdapter, 'step', step)
    path = tmp_path / 'recorded.rrd'
    args = [] if arm == 'both' else ['--arm', arm]  # Exercise the new default.
    assert main([*args, '--no-viewer', '--fast', '--steps', '530', '--save', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    rows = result['submissions']
    assert len(steps_checked) == 530
    assert [(row['source_frame_index'], row['arm']) for row in rows] == [
        (index, side) for index in range(64) for side in arms]
    assert len({row['chunk_id'] for row in rows}) == 2 * len(arms)
    for row in rows:
        source_time = trajectories[row['arm']].source_times[row['source_frame_index']]
        assert row['source_time'] == source_time
        assert 0 <= row['host_time'] - source_time < .00401
        assert row['gripper_opening'] == trajectories[row['arm']].gripper_openings[row['source_frame_index']]
        assert row['gripper_sequence'] == row['sequence'] + 1
    if arm == 'both':
        for left, right in zip(rows[::2], rows[1::2], strict=True):
            assert left['host_time'] == right['host_time']
            assert left['source_time'] == right['source_time']
    assert result['reference_source']['source_column'] == 'action'
    assert result['reference_source']['selected_arms'] == list(arms)
    assert result['display']['errors'] == 0 and result['display_closed']
    assert result['hardware_connections'] == 0
    assert all(result['joint_response_by_arm'][side] > 0 for side in arms)
    for joint in result['initial_joints']:
        if not any(joint.startswith(side + '_') for side in arms):
            assert result['final_joints'][joint] == result['initial_joints'][joint]
