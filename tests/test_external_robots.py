"""Cross-model behavior and actual libpyflexiv API-shaped mapping tests.

Set RMW_RIZON4_URDF to test the external licensed asset in addition to the
portable fixture. No vendor SDK is imported and no hardware can be opened.
"""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion_workbench import JointTarget, RobotSession, SampleTime, StreamEvent, TcpTarget
from robot_motion_workbench.adapters.flexiv import FlexivJointAdapter
from robot_motion_workbench.adapters.reference import (
    ReferenceRobotAdapter, demo_urdf, description_from_urdf, urdf_joint_controls,
)
from robot_motion_workbench.examples.callback_sdk import make_callback_session
from robot_motion_workbench.examples.reference import goal_positions, run_reference
from robot_motion_workbench.urdf import URDFModel


@pytest.fixture(params=['portable', 'rizon4'])
def description(request):
    path = demo_urdf()
    if request.param == 'rizon4':
        external = os.environ.get('RMW_RIZON4_URDF')
        if external is None:
            pytest.skip('Set RMW_RIZON4_URDF for the external licensed Rizon4 acceptance')
        path = Path(external).resolve(strict=True)
    return description_from_urdf(path)


def rotation_angle(a, b):
    return float(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)))


def test_every_axis_moves_independently_and_j7_rotates_flange(description):
    assert description.dof == 7
    assert not np.allclose(description.base_transform, np.eye(4))
    tool = description.tool_frames['tool']
    assert not np.allclose(tool.link_from_tool, np.eye(4))
    for index, joint in enumerate(description.joint_names):
        adapter = ReferenceRobotAdapter(description)
        session = RobotSession(adapter)
        initial = session.snapshot()
        q = dict(initial.joint_positions)
        q[joint] += .2
        inputs = session.activate('test')
        receipt = inputs.submit(JointTarget('arm', dict(reversed(list(q.items())))))
        assert receipt.status == 'submitted'
        assert session.snapshot().joint_positions == initial.joint_positions
        adapter.advance(.4)
        final = session.snapshot()
        for n in description.joint_names:
            assert final.joint_positions[n] - initial.joint_positions[n] == pytest.approx(.2 if n == joint else 0)
        first = adapter.model.link_transforms(initial.joint_positions)
        last = adapter.model.link_transforms(final.joint_positions)
        moving = adapter.model.joints[joint]
        np.testing.assert_allclose(first[moving.parent], last[moving.parent], atol=1e-14)
        assert rotation_angle(first[moving.child], last[moving.child]) == pytest.approx(.2)
        assert rotation_angle(first[tool.parent_link], last[tool.parent_link]) == pytest.approx(.2)
        np.testing.assert_allclose(final.tcp_poses['tool'], last[tool.parent_link] @ tool.link_from_tool)
        if index == 6:
            # J7 rotates the physical flange, even when its origin barely moves.
            assert rotation_angle(initial.tcp_poses['tool'], final.tcp_poses['tool']) == pytest.approx(.2)
            assert np.linalg.norm(final.tcp_poses['tool'][:3, 3] - initial.tcp_poses['tool'][:3, 3]) > .005
            if 'rizon4' in description.urdf_path.name:
                np.testing.assert_allclose(first[tool.parent_link][:3, 3], last[tool.parent_link][:3, 3], atol=1e-14)
        inputs.close()


def test_base_and_tool_are_composed_in_world(description):
    adapter = ReferenceRobotAdapter(description)
    q = adapter.snapshot().joint_positions
    world = adapter.model.link_transforms(q)
    local = adapter.model.link_transforms(q, np.eye(4))
    for n in local:
        np.testing.assert_allclose(world[n], description.base_transform @ local[n], atol=1e-14)
    tool = description.tool_frames['tool']
    np.testing.assert_allclose(adapter.snapshot().tcp_poses['tool'],
        description.base_transform @ local[tool.parent_link] @ tool.link_from_tool, atol=1e-14)


def test_policy_executes_all_seven_and_uses_no_ik(description):
    adapter = ReferenceRobotAdapter(description)
    assert not adapter.capabilities().tcp_targets
    before = adapter.snapshot()
    with pytest.raises(ValueError, match='Unsupported TCP'):
        RobotSession(adapter).activate('invalid').submit(TcpTarget('tool', np.eye(4)))
    result = run_reference(adapter, steps=36)
    assert len(result['submissions']) == 12
    assert result['sent'].startswith('unavailable')
    assert result['planned'].startswith('unavailable')
    for n in description.joint_names:
        assert abs(result['final_joints'][n] - before.joint_positions[n]) > .2


class SpyJointImpedanceControl:
    """Only the methods/fields exposed by actual flexiv_rt.cpp bindings.

    PyJointImpedanceControl: set_target_joints(positions, velocities=[],
    accelerations=[]), get_state(), is_running(). JointState.q is list[7].
    Calls are recorded; target acceptance deliberately never changes feedback.
    """
    def __init__(self):
        self.calls = []
        self.running = True
        self.q = np.arange(7, dtype=float) * .1
        self.error = None

    def is_running(self):
        self.calls.append(('is_running',))
        return self.running

    def get_state(self):
        self.calls.append(('get_state',))
        return SimpleNamespace(q=self.q, dq=[0.] * 7, tau=[0.] * 7,
                               tau_ext=[0.] * 7, tcp_pose=[0., 0., 0., 1., 0., 0., 0.])

    def set_target_joints(self, positions, velocities=(), accelerations=()):
        if self.error:
            raise self.error
        self.calls.append(('set_target_joints', positions.copy(), list(velocities), list(accelerations)))


def test_injected_flexiv_mapping_and_host_poll_time(description):
    sdk = SpyJointImpedanceControl()
    sdk_order = tuple(reversed(description.joint_names))
    clock = iter([100., 100.25, 100.5])
    adapter = FlexivJointAdapter(sdk, description, sdk_joint_names=sdk_order,
        joint_controls=urdf_joint_controls(description), clock=lambda: next(clock))
    assert sdk.calls == []  # No SDK access, connection, enable, mode or start.
    session = RobotSession(adapter)
    observed = session.snapshot()
    assert observed.time == .25
    timing = SampleTime(observed.time)
    assert timing.source_time is timing.source_clock is None
    assert not observed.tcp_poses  # SDK tcp_pose frame/tool calibration is unspecified.
    np.testing.assert_allclose([observed.joint_positions[n] for n in sdk_order], sdk.q)
    inputs = session.activate('sdk-test')
    goal = {n: -.05 * i for i, n in enumerate(description.joint_names)}
    receipt = inputs.submit(JointTarget('arm', goal))
    assert sdk.calls[-1] == ('set_target_joints', [goal[n] for n in sdk_order], [], [])
    assert receipt.status == 'submitted'
    assert session.snapshot().joint_positions == observed.joint_positions
    sdk.q[:] = 1.  # Snapshot is copied, never a live SDK buffer.
    assert observed.joint_positions[sdk_order[0]] == 0.
    assert set(c[0] for c in sdk.calls) == {'get_state', 'set_target_joints', 'is_running'}


def test_injected_adapter_rejects_bad_mapping_stopped_control_and_bad_feedback(description):
    sdk = SpyJointImpedanceControl()
    controls = urdf_joint_controls(description)
    names = description.joint_names
    for order in [names[:-1], names[:-1] + names[:1]]:
        with pytest.raises(ValueError, match='seven unique'):
            FlexivJointAdapter(sdk, description, sdk_joint_names=order, joint_controls=controls)
    adapter = FlexivJointAdapter(sdk, description, sdk_joint_names=names, joint_controls=controls)
    session = RobotSession(adapter)
    inputs = session.activate('sdk-test')
    invalid = dict.fromkeys(names[:-1], 0.)
    with pytest.raises(ValueError, match='exactly match'):
        inputs.submit(JointTarget('arm', invalid))
    assert sdk.calls == []
    sdk.running = False
    with pytest.raises(RuntimeError, match='not running'):
        inputs.submit(JointTarget('arm', dict.fromkeys(names, 0.)))
    with pytest.raises(RuntimeError, match='not running'):
        session.snapshot()
    assert session.last_receipt is None
    sdk.running = True
    for q in [[0.] * 6, [float('nan')] * 7]:
        sdk.q = q
        with pytest.raises(ValueError, match='seven finite'):
            session.snapshot()
    sdk.error = RuntimeError('SDK target jump rejected')
    with pytest.raises(RuntimeError, match='jump rejected'):
        inputs.submit(JointTarget('arm', dict.fromkeys(names, 0.)))
    assert session.last_receipt is None


def test_callback_example_maps_an_arbitrary_sdk_order(description):
    order = description.joint_names[2:] + description.joint_names[:2]
    q = np.linspace(0., .3, len(order))
    writes = []
    session = make_callback_session(description, urdf_joint_controls(description), sdk_joint_names=order,
                                    read_positions=lambda: q, write_positions=writes.append)
    initial = session.snapshot()
    target = {n: .01 * (i + 1) for i, n in enumerate(description.joint_names)}
    inputs = session.activate('callback-test')
    inputs.submit(JointTarget('arm', target))
    assert writes == [[target[n] for n in order]]
    assert session.snapshot().joint_positions == initial.joint_positions
    with pytest.raises(ValueError, match='exactly once'):
        make_callback_session(description, urdf_joint_controls(description), sdk_joint_names=order[:-1],
                              read_positions=lambda: q, write_positions=writes.append)


def test_reference_imports_have_no_optional_runtime_side_effects():
    subprocess.run([sys.executable, '-c', '''import sys
import robot_motion_workbench
import robot_motion_workbench.adapters.reference
import robot_motion_workbench.adapters.flexiv
import robot_motion_workbench.examples.reference
import robot_motion_workbench.examples.callback_sdk
assert not {'rerun', 'PySide6', 'trimesh', 'pinocchio', 'libpyflexiv', 'libpyelite', 'flexiv_rt'} & sys.modules.keys()
'''], check=True)


def test_host_poll_state_time_does_not_become_source_sample_time():
    from robot_motion_workbench.policy_renderer import event_data
    description = description_from_urdf()
    sdk = SpyJointImpedanceControl()
    clock = iter([20., 20.75])
    adapter = FlexivJointAdapter(sdk, description, sdk_joint_names=description.joint_names,
        joint_controls=urdf_joint_controls(description), clock=lambda: next(clock))
    state = adapter.snapshot()
    data = event_data(StreamEvent('feedback', SampleTime(state.time), 'measured', state=state,
                                  reason='Synthetic API spy, no hardware'))
    assert data['feedback']['state_time'] == data['timing']['received_at'] == .75
    assert data['timing']['source_time'] is data['timing']['source_clock'] is None
    assert 'source_state_time' not in data['feedback']
