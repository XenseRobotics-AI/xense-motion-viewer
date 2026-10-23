"""Subprocess offscreen integration probe; launched with the wheel's Qt ABI."""
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from robot_motion_workbench import JointTarget, RobotSession
from robot_motion_workbench.qt import ControlPanel
from test_control import FakeSevenAxis


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    urdf = directory / 'fake.urdf'
    urdf.write_text('<robot name="fake"><link name="root"/></robot>')
    app = QApplication.instance() or QApplication([])
    sdk = FakeSevenAxis(urdf)
    controller = RobotSession(sdk.adapter)
    panel = ControlPanel(controller.activate('qt'))
    panel.resize(850, 900)
    panel.show()
    app.processEvents()
    assert len(panel.joint_editors['arm']) == 7
    assert len(panel.joint_editors['wrist']) == 3
    assert list(panel.tcp_editors) == ['gripper']
    assert sdk.accepted == []
    before = controller.snapshot()
    # Exercise actual Qt editing/click signals and reversed mm mapping.
    slider_axis = panel.joint_editors['arm']['axis/1']
    assert slider_axis.suffix() == ' mm'
    slider_axis.setValue(-115)  # (-115 - 10) / -1000 = .125 m
    panel.joint_editors['arm']['axis/6'].setValue(30)
    assert sdk.accepted == []
    QTest.mouseClick(panel.joint_buttons['arm'], Qt.MouseButton.LeftButton)
    assert sdk.accepted[-1].positions['axis/1'] == .125
    np.testing.assert_allclose(sdk.accepted[-1].positions['axis/6'], np.pi / 6)
    assert tuple(sdk.accepted[-1].positions) == sdk.order
    assert panel.last_receipt.status == 'submitted'
    assert 'sent/achieved unknown' in panel.status.text()
    assert dict(controller.snapshot().joint_positions) == dict(before.joint_positions)
    sdk.q['axis/1'] = .08
    for _ in range(3):
        panel.refresh()
        app.processEvents()
    assert len(sdk.accepted) == 1
    assert slider_axis.value() == -115  # observations cannot overwrite targets
    assert panel.observed_joints['arm']['axis/1'].text() == '-70.000000 mm'
    panel.grab().save(str(directory / 'fake-seven-axis-joints.png'))
    panel.tabs.setCurrentIndex(1)
    panel.tcp_editors['gripper']['x'].setValue(.023)
    panel.tcp_editors['gripper']['yaw'].setValue(.2)
    QTest.mouseClick(panel.tcp_buttons['gripper'], Qt.MouseButton.LeftButton)
    assert len(sdk.accepted) == 2
    assert sdk.accepted[-1].world_from_tcp[0, 3] == .023
    np.testing.assert_array_equal(controller.snapshot().tcp_poses['gripper'], np.eye(4))
    panel.grab().save(str(directory / 'fake-seven-axis-cartesian.png'))
    QTest.mouseClick(panel.sync_button, Qt.MouseButton.LeftButton)
    assert len(sdk.accepted) == 2 and slider_axis.value() == -70
    # A different producer takes over through the exact same public API.
    policy = controller.activate('policy', replace=True)
    policy.submit(JointTarget('arm', sdk.q))
    panel.refresh()
    assert not panel.joint_buttons['arm'].isEnabled()
    assert not panel.tcp_buttons['gripper'].isEnabled()
    assert panel.submit_joints('arm') is None
    assert len(sdk.accepted) == 3 and controller.last_receipt.source == 'policy'
    panel.close()
    # Capability absence removes Cartesian controls entirely.
    joint_sdk = FakeSevenAxis(urdf, cartesian=False)
    joint_panel = ControlPanel(RobotSession(joint_sdk.adapter).activate('qt'))
    assert joint_panel.tabs.count() == 1 and not joint_panel.tcp_editors
    joint_panel.show(); app.processEvents()
    joint_panel.grab().save(str(directory / 'joint-only.png'))
    joint_panel.close()
    assert joint_sdk.accepted == []
    result = dict(custom_axes=7, named_subgroup_axes=3, submitted=3,
                  mapping='negative-scale mm + offset and radians/degrees passed',
                  observations_did_not_submit=True, targets_did_not_move_state=True,
                  explicit_ownership=True, capability_based_cartesian=True)
    (directory / 'qt-results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
