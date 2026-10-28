"""Exercise the retained production sliders through the common command API."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import rerun as rr
from libpyelite.sim import StationSimulation
from PySide6.QtWidgets import QTabWidget
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QStyle, QStyleOptionSlider
from PySide6.QtTest import QTest

from xense_motion_viewer import JointTarget, TcpTarget
from xense_motion_viewer.adapters.elite import EliteRerunView, make_qt_window


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sim = StationSimulation(collision_profile='auto')
    recording = rr.RecordingStream('elite-common-inputs')
    recording.save(directory / 'elite-inputs.rrd')
    view = EliteRerunView(sim, recording)
    accepted = []
    original = view.adapter.submit
    def submit(command):
        original(command)
        accepted.append(command)
    view.adapter.submit = submit
    app, window = make_qt_window(sim, view)
    window.render_timer.stop()
    window.show()
    app.processEvents()
    assert accepted == []
    before = sim.q.copy()
    control = window.left_joints.controls[0]
    control.slider.setValue(control.slider.value() + 1)
    assert isinstance(accepted[-1], JointTarget)
    assert window.session.last_receipt.source == 'qt'
    assert 'sent/achieved unknown' in window.receipt_status.text()
    np.testing.assert_array_equal(sim.q, before)
    window.advance(4)
    assert len(accepted) == 1
    window.grab().save(str(directory / 'elite-slider-joints.png'))
    tabs = window.findChild(QTabWidget)
    tabs.setCurrentIndex(1)
    control = window.left_pose.controls['x']
    control.slider.setValue(control.slider.value() + 1)
    assert isinstance(accepted[-1], TcpTarget)
    window.advance(4)
    assert len(accepted) == 2
    app.processEvents()
    window.grab().save(str(directory / 'elite-slider-cartesian.png'))
    # Readable original diagnostic text remains, never raw planner JSON.
    assert 'QP failure=' in window.ik_status.text()
    assert 'collision=' in window.ik_status.text()
    tabs.setCurrentIndex(0)
    for side in sim.arms:
        group = getattr(window, side + '_joints')
        assert window.gripper_panel.rows[side].parentWidget() is group.widget
        assert group.widget.layout().indexOf(window.gripper_panel.rows[side]) == 6
        group = getattr(window, side + '_pose')
        assert window.ik_gripper_panel.rows[side].parentWidget() is group.widget
        assert group.widget.layout().indexOf(window.ik_gripper_panel.rows[side]) == 6
    for side, value in [('left', 200), ('right', 650)]:
        window.gripper_panel.sliders[side].setValue(value)
        assert accepted[-1].group == side + '_gripper'
        assert window.ik_gripper_panel.sliders[side].value() == value
        assert window.ik_gripper_panel.editors[side].value() == value / 10
    assert len(accepted) == 4
    window.advance(130)
    assert view.adapter.grippers.opening('left') == .2
    assert view.adapter.grippers.opening('right') == .65
    assert len(accepted) == 4  # Observation refresh must not resubmit targets.
    app.processEvents()
    window.grab().save(str(directory / 'elite-gripper-sliders.png'))
    tabs.setCurrentIndex(1)
    for side, value in [('left', 40), ('right', 55)]:
        window.ik_gripper_panel.editors[side].setValue(value)
        assert accepted[-1].group == side + '_gripper'
        assert window.gripper_panel.sliders[side].value() == value * 10
        assert window.gripper_panel.editors[side].value() == value
    assert len(accepted) == 6  # One submission per edit, including mirrored targets.
    window.gripper_panel.refresh()
    assert window.gripper_panel.editors['left'].value() == 40  # Still moving: preserve target.
    assert window.ik_gripper_panel.editors['left'].value() == 40
    for index in [0, 1]:
        tabs.setCurrentIndex(index)
        app.processEvents()
    assert len(accepted) == 6
    window.advance(130)
    for side, value in [('left', .4), ('right', .55)]:
        assert view.adapter.grippers.opening(side) == value
        assert window.gripper_panel.observed[side].text() == window.ik_gripper_panel.observed[side].text()
    assert len(accepted) == 6
    window.grab().save(str(directory / 'elite-ik-gripper-sliders.png'))
    # Use actual mouse events and the real timer, not just setValue + advance.
    window.render_timer.start()
    for side in sim.arms:
        slider = window.ik_gripper_panel.sliders[side]
        option = QStyleOptionSlider(); slider.initStyleOption(option)
        handle = slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, option,
                                               QStyle.SubControl.SC_SliderHandle, slider)
        target = QPoint(slider.width() - handle.width() // 2 - 1, handle.center().y())
        QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=handle.center())
        QTest.mouseMove(slider, target, delay=20)
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=target)
    QTest.qWait(900)
    window.render_timer.stop()
    for side in sim.arms:
        assert view.adapter.grippers.opening(side) > .98
        expected = window.gripper_panel.sliders[side].value() / 1000
        assert window.ik_gripper_panel.sliders[side].value() == round(expected * 1000)
        assert abs(view.adapter.grippers.opening(side) - expected) < 1e-6
    # The initial mesh upload can still be in flight; let the bounded display
    # worker consume the final finger state without stepping the simulation.
    for _ in range(100):
        if view.stats.rendered_time == view.stats.latest_time:
            break
        time.sleep(.02)  # Release the GIL for the Python display worker.
    assert view.stats.errors == 0 and view.stats.rendered_time > .5, view.stats
    accepted_count = len(accepted)
    policy = window.session.activate('policy', replace=True)
    pose = sim.tcp_poses()['left'].copy(); pose[0, 3] += .001
    policy.submit(TcpTarget('left', pose))
    assert len(accepted) == accepted_count + 1
    control.slider.setValue(control.slider.value() + 1)
    assert len(accepted) == accepted_count + 1
    assert 'rejected' in window.receipt_status.text()
    window.gripper_panel.sliders['left'].setValue(800)
    assert len(accepted) == accepted_count + 1
    assert '拒绝' in window.gripper_panel.statuses['left'].text()
    window.ik_gripper_panel.sliders['right'].setValue(700)
    assert len(accepted) == accepted_count + 1
    assert '拒绝' in window.ik_gripper_panel.statuses['right'].text()
    snapshot = sim.q.copy()
    window._reset_home()
    np.testing.assert_array_equal(sim.q, snapshot)  # old Qt lease cannot reset policy state
    window.advance(4)
    assert len(accepted) == accepted_count + 1
    for panel in (window.gripper_panel, window.ik_gripper_panel):
        assert all(not slider.isEnabled() for slider in panel.sliders.values())
        assert all(not editor.isEnabled() for editor in panel.editors.values())
    window.close()
    assert window.session.active_source == 'policy'  # stale Qt close cannot release policy
    view.renderer.flush(); recording.disconnect()
    result = dict(original_sliders=True, joint_and_tcp_via_common_api=True,
                  observation_refresh_does_not_submit=True, readable_diagnostics=True,
                  ownership_rejects_stale_qt=True, gripper_sliders=True, real_mouse_drag=True,
                  ik_gripper_sliders=True, synchronized_gripper_targets=True,
                  submitted=len(accepted))
    (directory / 'elite-qt-results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
