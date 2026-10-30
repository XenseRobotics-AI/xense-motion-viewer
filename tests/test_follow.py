import copy
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion_workbench import InputOwnershipError, JointTarget, TcpTarget
from robot_motion_workbench.follow import FollowInput, FollowTrace, QtInputRoutes, command_data, rotation_vector


class Input:
    active = True

    def __init__(self):
        self.commands = []
        self.on_submit = None

    def submit(self, command):
        if not self.active:
            raise InputOwnershipError('inactive')
        self.commands.append(command)
        receipt = SimpleNamespace(command=command, sequence=len(self.commands), source='follow')
        if self.on_submit:
            self.on_submit(receipt)
        return receipt


def engine():
    source = Input()
    poses = {side: np.eye(4) for side in ('left', 'right')}
    poses['left'][:3, 3] = [.6, .3, .8]
    poses['right'][:3, 3] = [.6, -.2, .8]
    follow = FollowInput(source, lambda side: poses[side])
    source.on_submit = follow.record_accepted
    follow.set_enabled(True)
    return follow, source, poses


def steps(follow, count):
    for _ in range(count):
        follow.before_step()
        follow.after_step({'q': [0.] * 12})


def test_hand_is_independent_release_holds_and_regrip_reanchors_only_target():
    follow, source, feedback = engine()
    follow.begin_grip('left')
    follow.move_hand('left', [.05, -.02, -.1], [0., .08, 0.])
    hand = follow.hands['left'].copy()
    target = follow.targets['left'].copy()
    steps(follow, 30)
    np.testing.assert_allclose(source.commands[-1].world_from_tcp[:3, 3], [.65, .28, .7])
    np.testing.assert_allclose(target[:3, :3], rotation_vector([0, .08, 0]))
    follow.end_grip('left')
    feedback['left'][2, 3] = .75  # Robot stopped above the commanded target.
    steps(follow, 250)
    np.testing.assert_array_equal(follow.hands['left'], hand)
    np.testing.assert_array_equal(source.commands[-1].world_from_tcp, target)
    np.testing.assert_array_equal(feedback['left'][:3, 3], [.6, .3, .75])
    follow.begin_grip('left')
    np.testing.assert_array_equal(follow.hands['left'], hand)
    np.testing.assert_array_equal(follow.targets['left'], feedback['left'])
    follow.move_hand('left', [0, 0, .02])
    np.testing.assert_allclose(follow.targets['left'][:3, 3], [.6, .3, .77])
    assert not any(c.tcp == 'right' for c in source.commands)


def test_targets_are_sampled_at_30hz_on_native_ticks_even_without_mouse_events():
    follow, source, _ = engine()
    follow.start_recording({}, [0.] * 12)
    follow.begin_grip('left')
    follow.begin_grip('right')
    steps(follow, 250)
    follow.stop_recording()
    assert len(source.commands) == 60
    times = [r['step'] for r in follow.trace.commands[::2]]
    assert times[:4] == [0, 9, 17, 25]
    assert set(np.diff(times)) == {8, 9}
    assert len(follow.trace.samples) == 250


def test_rotation_can_run_concurrently_with_translation():
    follow, source, _ = engine()
    follow.begin_grip('left')
    follow.angular_rates['left'][1] = .3
    for _ in range(100):
        follow.move_hand('left', [0, 0, -.0001])
        steps(follow, 1)
    np.testing.assert_allclose(follow.hands['left'][:3, 3], [0, 0, -.01])
    np.testing.assert_allclose(follow.hands['left'][:3, :3], rotation_vector([0, .12, 0]), atol=1e-12)
    follow.end_grip('left')
    hand = follow.hands['left'].copy()
    steps(follow, 100)
    np.testing.assert_array_equal(follow.hands['left'], hand)


def test_record_replay_preserves_command_order_steps_hands_and_grippers(tmp_path):
    original, source, _ = engine()
    original.start_recording({'build': 'A'}, [0.] * 12)
    original.begin_grip('left')
    for i in range(100):
        original.move_hand('left', [0, 0, -.0001])
        if i == 23:
            source.submit(JointTarget('left_gripper', {'finger': .2}))
        steps(original, 1)
    original.end_grip('left')
    steps(original, 20)
    original.stop_recording()
    trace = original.trace
    path = tmp_path / 'follow.json'
    trace.save(path)
    loaded = FollowTrace.load(path)
    replay, output, feedback = engine()
    feedback['left'][0, 3] += .05  # Replay uses recorded goals, never a fresh grip mapping.
    replay.start_playback(loaded, metadata={'build': 'B'})
    steps(replay, loaded.duration_steps + 1)
    assert replay.playback is None
    assert [command_data(c) for c in output.commands] == [command_data(c) for c in source.commands]
    assert replay.trace.commands == loaded.commands
    assert replay.trace.samples == loaded.samples
    assert replay.trace.metadata['build'] == 'B'
    assert replay.trace.end_reason == 'completed'


@pytest.mark.parametrize('mutation', ['negative_step', 'unordered', 'nan_pose', 'duration', 'reflection'])
def test_invalid_recordings_are_rejected(mutation):
    follow, _, _ = engine()
    follow.start_recording({}, [0.] * 12)
    follow.begin_grip('left')
    steps(follow, 30)
    follow.stop_recording()
    trace = copy.deepcopy(follow.trace)
    if mutation == 'negative_step': trace.commands[0]['step'] = -1
    if mutation == 'unordered': trace.commands.reverse()
    if mutation == 'nan_pose': trace.commands[0]['command']['pose'][0][3] = float('nan')
    if mutation == 'duration': trace.duration_steps = 1_000_000
    if mutation == 'reflection': trace.commands[0]['command']['pose'][0][0] = -1
    with pytest.raises(ValueError): trace.validate()


def test_fault_or_external_takeover_stops_follow_submission_and_capture():
    follow, source, _ = engine()
    follow.start_recording({}, [0.] * 12)
    follow.begin_grip('left')
    steps(follow, 20)
    follow.fault('native collision latch')
    count = len(source.commands)
    steps(follow, 50)
    assert len(source.commands) == count and follow.recording is None
    assert 'fault' in follow.trace.end_reason
    with pytest.raises(InputOwnershipError): follow.begin_grip('left')
    follow.reset()
    follow.start_playback(follow.trace)
    steps(follow, 2)
    source.active = False
    steps(follow, 50)
    assert follow.playback is None and follow.play_capture is None
    assert follow.trace.end_reason == 'input ownership lost'


def test_routes_switch_only_their_own_lease(robot_urdf):
    from test_control import FakeSevenAxis
    from robot_motion_workbench import RobotSession
    sdk = FakeSevenAxis(robot_urdf)
    session = RobotSession(sdk.adapter)
    routes = QtInputRoutes(session)
    target = JointTarget('arm', sdk.q)
    routes.legacy.submit(target)
    routes.select('follow')
    assert routes.follow.active and not routes.legacy.active
    with pytest.raises(InputOwnershipError): routes.legacy.submit(target)
    routes.follow.submit(target)
    assert session.last_receipt.source == 'follow'
    session.activate('policy', replace=True)
    routes.select('qt')
    routes.select('follow')
    assert not routes.follow.active and not routes.legacy.active
    routes.close()
    assert session.active_source == 'policy'
