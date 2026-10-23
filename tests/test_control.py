from dataclasses import FrozenInstanceError
import importlib.abc
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robot_motion_workbench import (
    CallbackRobotAdapter, ControlCapabilities, InputOwnershipError, JointControl,
    JointTarget, RobotDescription, RobotSession, RobotState, TcpControl, TcpTarget,
)


class FakeSevenAxis:
    """Independent SDK-shaped callback implementation; accepts but never moves."""
    def __init__(self, urdf, *, cartesian=True):
        self.names = tuple(f'axis/{i}' for i in range(7))
        self.order = (self.names[6], *self.names[:6])
        self.q = dict.fromkeys(self.names, 0.0)
        self.pose = np.eye(4)
        self.accepted = []
        self.reject = False
        self.description = RobotDescription('external-7', urdf, self.names)
        self.controls = ControlCapabilities(
            joints={n: (JointControl('m', -0.5, 0.5, 'mm', -1000, 10) if i == 1
                        else JointControl('rad', -3, 3, 'deg', 180 / np.pi))
                    for i, n in enumerate(self.names)},
            groups={'arm': self.order, 'wrist': self.names[4:]},
            tcp_targets={'gripper': TcpControl('arm')} if cartesian else {},
        )
        self.adapter = CallbackRobotAdapter(self.description, self.controls,
            read_state=self.read_state, submit_target=self.accept)

    def accept(self, command):
        if self.reject:
            raise RuntimeError('SDK target refused')
        self.accepted.append(command)

    def read_state(self):
        return RobotState(0.0, self.q, tcp_poses={'gripper': self.pose})


def test_seven_axis_reorder_receipt_observations_and_named_subgroup(robot_urdf):
    sdk = FakeSevenAxis(robot_urdf)
    controller = RobotSession(sdk.adapter)
    producer = controller.activate('policy')
    values = {n: i / 100 for i, n in enumerate(sdk.names)}
    before = controller.snapshot()
    receipt = producer.submit(JointTarget('arm', dict(reversed(list(values.items())))))
    assert receipt.status == 'submitted' and receipt.source == 'policy' and receipt.sequence == 1
    assert tuple(sdk.accepted[-1].positions) == sdk.order
    assert dict(sdk.accepted[-1].positions) == values
    assert dict(controller.snapshot().joint_positions) == dict(before.joint_positions)
    assert not hasattr(receipt, 'sent') and not hasattr(receipt, 'achieved')
    assert controller.last_receipt is receipt
    producer.submit(JointTarget('wrist', {n: values[n] for n in sdk.names[4:]}))
    pose = np.eye(4); pose[0, 3] = .02
    producer.submit(TcpTarget('gripper', pose))
    np.testing.assert_array_equal(controller.snapshot().tcp_poses['gripper'], np.eye(4))
    sdk.q[sdk.names[0]] = .012  # independent later SDK observation
    assert controller.snapshot().joint_positions[sdk.names[0]] == .012
    assert len(sdk.accepted) == 3


def test_invalid_inputs_never_reach_sdk(robot_urdf):
    sdk = FakeSevenAxis(robot_urdf)
    producer = RobotSession(sdk.adapter).activate('programmatic')
    values = dict(sdk.q)
    for command in (JointTarget('unknown', values), JointTarget('arm', values | {'typo': 0}),
                    JointTarget('arm', {sdk.names[0]: 0}),
                    JointTarget('arm', values | {sdk.names[1]: .51}), TcpTarget('missing', np.eye(4))):
        with pytest.raises(ValueError):
            producer.submit(command)
    for pose in (np.zeros((4, 4)), np.eye(3), np.eye(4) * np.nan,
                 np.diag([-1., 1, 1, 1]), np.diag([2., 1, 1, 1])):
        with pytest.raises(ValueError):
            producer.submit(TcpTarget('gripper', pose))
    for value in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            producer.submit(JointTarget('arm', values | {sdk.names[0]: value}))
    with pytest.raises(TypeError):
        producer.submit(np.zeros(7))
    assert sdk.accepted == []
    assert producer.controller.last_receipt is None


def test_command_metadata_and_receipt_immutability(robot_urdf):
    sdk = FakeSevenAxis(robot_urdf)
    values = dict(sdk.q)
    command = JointTarget('arm', values)
    values[sdk.names[0]] = 2
    assert command.positions[sdk.names[0]] == 0
    with pytest.raises(TypeError):
        command.positions[sdk.names[0]] = 1
    pose = np.eye(4)
    target = TcpTarget('gripper', pose)
    pose[0, 3] = 1
    assert target.world_from_tcp[0, 3] == 0
    with pytest.raises(ValueError):
        target.world_from_tcp.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        target.tcp = 'new'
    with pytest.raises(TypeError):
        sdk.controls.groups['new'] = sdk.names
    receipt = RobotSession(sdk.adapter).activate('policy').submit(command)
    with pytest.raises(FrozenInstanceError):
        receipt.status = 'achieved'


def test_input_ownership_takeover_release_and_sdk_failure(robot_urdf):
    sdk = FakeSevenAxis(robot_urdf)
    controller = RobotSession(sdk.adapter)
    qt = controller.activate('qt')
    with pytest.raises(InputOwnershipError):
        controller.activate('policy')
    policy = controller.activate('policy', replace=True)
    target = JointTarget('arm', sdk.q)
    with pytest.raises(InputOwnershipError):
        qt.submit(target)
    qt.close()  # stale close must not release a newer owner
    assert controller.active_source == 'policy'
    receipt = policy.submit(target)
    sdk.reject = True
    with pytest.raises(RuntimeError, match='SDK target refused'):
        policy.submit(target)
    assert controller.last_receipt is receipt and len(sdk.accepted) == 1
    policy.close()
    assert controller.active_source is None
    again = controller.activate('policy')
    assert again.active and not policy.active
    with pytest.raises(InputOwnershipError):
        policy.submit(target)


def test_metadata_consistency_and_no_cartesian(robot_urdf):
    sdk = FakeSevenAxis(robot_urdf, cartesian=False)
    controller = RobotSession(sdk.adapter)
    with pytest.raises(ValueError, match='Unsupported TCP'):
        controller.activate('programmatic').submit(TcpTarget('gripper', np.eye(4)))
    for groups in ({'arm': ('unknown',)}, {'arm': ('axis/0', 'axis/0')}, {'arm': ()}):
        with pytest.raises(ValueError):
            ControlCapabilities(sdk.controls.joints, groups)
    sdk.adapter._capabilities = ControlCapabilities({}, {})
    with pytest.raises(ValueError, match='exactly'):
        RobotSession(sdk.adapter)
    sdk.q.pop('axis/0')
    with pytest.raises(ValueError, match='exactly'):
        controller.snapshot()


def test_core_imports_without_optional_dependencies():
    code = '''
import importlib.abc, sys
blocked = {'PySide6', 'rerun', 'libpyelite', 'pinocchio', 'trimesh', 'yaml'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('blocked ' + fullname, name=fullname)
sys.meta_path.insert(0, Block())
from robot_motion_workbench import RobotSession, JointTarget, TcpTarget, CallbackRobotAdapter
from robot_motion_workbench.adapters.elite import _simulation
from types import SimpleNamespace
assert not blocked.intersection(sys.modules)
try:
    _simulation(SimpleNamespace())
except RuntimeError as exc:
    assert 'current-production-source' in str(exc) and 'README.md' in str(exc)
    assert isinstance(exc.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('missing Elite should fail with installation diagnostic')
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
