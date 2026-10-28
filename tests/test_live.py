"""Live display contracts verified through actual Rerun recordings."""
from dataclasses import replace

import numpy as np
import pytest
import rerun as rr
from rrd_reader import RrdReader, clear_times

from xense_motion_viewer import RobotDescription, RobotState, poses_from_xyz_rotation6d
from xense_motion_viewer.live import FutureTrajectory, LiveFrame, LiveMotionDisplay, LiveRerunRenderer


def rows(chunks, path, component):
    return [value for chunk in chunks if chunk.entity_path == '/' + path
            for batch in [chunk.to_record_batch()] if component in batch.schema.names
            for value in batch.column(component).to_pylist() if value is not None]


def trajectory(ids, xs):
    poses = np.tile(np.eye(4), (len(ids), 1, 1))
    poses[:, 0, 3] = xs
    return FutureTrajectory(tuple(ids), poses)


def test_live_futures_progress_replace_and_measured_history(tmp_path, robot_urdf, names):
    rr.set_strict_mode(True)
    recording = rr.RecordingStream('live-test')
    path = tmp_path / 'live.rrd'
    recording.save(path)
    renderer = LiveRerunRenderer(recording, RobotDescription('live', robot_urdf, names))
    future = trajectory(['a', 'b', 'c'], [1., 2., 3.])
    state = RobotState(0, dict.fromkeys(names, 0.), tcp_poses={'left': np.eye(4)})
    renderer.consume_batch((LiveFrame(0, {'left': future}, state, 'sample-1'),))
    # Prediction points do not become measured history. Repeated SDK samples
    # do not become additional history points, either.
    remaining = FutureTrajectory(future.step_ids[1:], future.poses[1:])
    renderer.consume_batch((LiveFrame(.1, {'left': remaining}, state, 'sample-1'),))
    renderer.consume_batch((LiveFrame(.2, {'left': trajectory(['new'], [7.])},
                                      replace(state, time=.2), 'sample-2'),))
    renderer.consume_batch((LiveFrame(.3, {}, None, status='disconnected'),))
    renderer.consume_batch((LiveFrame(.4, {}, state, 'sample-3'),))
    assert renderer.robot.update_count == 3
    renderer.close()
    chunks = list(RrdReader(path).stream())
    root = renderer.root
    points = rows(chunks, f'{root}/future/left/steps', 'Points3D:positions')
    assert len(points) == 2  # Coordinates only on arrival/replacement.
    np.testing.assert_allclose(points[0], [[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    radii = rows(chunks, f'{root}/future/left/steps', 'Points3D:radii')
    assert any(len(value) == 3 and value[0] == 0 and value[1] > 0 for value in radii)
    history = rows(chunks, f'{root}/measured/left/history/0', 'LineStrips3D:strips')
    assert len(history) == 2
    measured = max(history, key=lambda value: len(value[0]))
    assert len(measured[0]) == 2
    np.testing.assert_allclose(measured[0], np.zeros((2, 3)))
    # The reconnect starts a new segment rather than bridging missing data.
    resumed = rows(chunks, f'{root}/measured/left/history/1', 'LineStrips3D:strips')
    assert len(resumed[0][0]) == 1
    assert clear_times(chunks, f'{root}/future/left', 'host_elapsed')
    assert clear_times(chunks, f'{root}/measured/left/current', 'host_elapsed')


def test_reset_and_async_close(tmp_path, robot_urdf, names):
    recording = rr.RecordingStream('live-reset')
    recording.save(tmp_path / 'reset.rrd')
    display = LiveMotionDisplay(recording, RobotDescription('live', robot_urdf, names))
    state = RobotState(0, dict.fromkeys(names, 0), tcp_poses={'tcp': np.eye(4)})
    assert display.publish(LiveFrame(0, state=state, sample_id='a'))
    assert display.publish(LiveFrame(.1, state=state, sample_id='a', reset=True))
    assert display.close()
    assert display.stats.errors == 0
    assert not display.stats.alive
    assert not display.publish(LiveFrame(.2))
    assert len(display.renderer._trails['tcp'].blocks[0]) == 1


def test_live_data_is_immutable_and_rotation_convention():
    values = np.array([[1, 2, 3, 0, 2, 0, -3, 1, 0.]])
    poses = poses_from_xyz_rotation6d(values)
    np.testing.assert_allclose(poses[0, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    future = FutureTrajectory(('step',), poses)
    poses[:] = 0
    assert future.poses[0, 0, 3] == 1
    with pytest.raises(ValueError):
        future.poses.setflags(write=True)
    with pytest.raises(ValueError, match='unique'):
        FutureTrajectory(('same', 'same'), np.tile(np.eye(4), (2, 1, 1)))
    with pytest.raises(ValueError, match='Degenerate'):
        poses_from_xyz_rotation6d(np.zeros((1, 9)))
