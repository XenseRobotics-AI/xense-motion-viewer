"""Exercise the real generic Qt signals on a supplied external URDF."""
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from robot_motion_workbench import InputOwnershipError, JointTarget, PolicyChunk, PolicyInput, RobotSession, SampleTime
from robot_motion_workbench.adapters.reference import ReferenceRobotAdapter, description_from_urdf
from robot_motion_workbench.examples.reference import goal_positions
from robot_motion_workbench.qt import ControlPanel

description = description_from_urdf(Path(sys.argv[1]))
adapter = ReferenceRobotAdapter(description)
session = RobotSession(adapter)
app = QApplication([])
qt = session.activate('qt')
panel = ControlPanel(qt)
assert len(panel.joint_editors['arm']) == 7
assert not panel.tcp_editors
assert session.last_receipt is None
initial = session.snapshot()
for n in description.joint_names:
    target = dict(initial.joint_positions)
    target[n] = goal_positions(adapter)[n]
    for key, value in target.items():
        panel.joint_editors['arm'][key].setValue(session.capabilities.joints[key].to_display(value))
    assert session.last_receipt is None or session.last_receipt.command.positions != target
    panel.joint_buttons['arm'].click()
    assert panel.last_receipt.source == 'qt'
    for key, value in target.items():
        assert abs(panel.last_receipt.command.positions[key] - value) < 1e-10
    assert session.snapshot().joint_positions == initial.joint_positions
    adapter.advance(.6)
    panel.refresh()
    assert abs(session.snapshot().joint_positions[n] - target[n]) < 1e-10
    # Reset explicitly via adapter-owned reference state for independent trials.
    qt.submit(JointTarget('arm', initial.joint_positions))
    adapter.advance(.6)
events = []
policy = PolicyInput(session.activate('policy', replace=True), events.append)
panel.refresh()
assert not panel.joint_buttons['arm'].isEnabled()
try:
    qt.submit(JointTarget('arm', initial.joint_positions))
    raise AssertionError('Old Qt lease submitted after policy takeover')
except InputOwnershipError:
    pass
now = adapter.time
values = np.array([[goal_positions(adapter)[n] for n in description.joint_names]])
chunk = PolicyChunk('qt-takeover', 'arm', description.joint_names, [.1], values,
                    SampleTime(now), now + 1., 'absolute_joint_positions', 'joint')
policy.accept(chunk)
policy.select(0, now=now)
submission = policy.execute_selected(now=now)
assert submission.receipt.source == 'policy'
assert session.snapshot().joint_positions == initial.joint_positions
adapter.advance(.6)
assert all(abs(session.snapshot().joint_positions[n] - initial.joint_positions[n]) > .2 for n in description.joint_names)
policy.close()
Path(sys.argv[2]).write_text(json.dumps(dict(axes=7, qt_axis_submissions=7, tcp_controls=0,
    policy_takeover=True, old_lease_rejected=True, observations_separate=True, hardware_connections=0), indent=2)+'\n')
