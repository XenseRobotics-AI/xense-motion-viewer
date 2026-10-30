"""Real Qt mouse/key input, native planning and deterministic Follow replay."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
from libpyelite.sim import StationSimulation
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QTabWidget

from robot_motion_workbench import TcpTarget
from robot_motion_workbench.adapters.elite import make_qt_window
from robot_motion_workbench.follow import FollowTrace


def main(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sim = StationSimulation(collision_profile='auto')
    app, window = make_qt_window(sim, None)
    window.render_timer.stop()
    window.show(); app.processEvents()
    tabs = window.findChild(QTabWidget)
    panel = window.follow_panel
    assert tabs.count() == 3 and tabs.widget(2) is panel
    accepted = []
    original = panel.adapter.submit
    def submit(command):
        original(command); accepted.append(command)
    panel.adapter.submit = submit
    tabs.setCurrentIndex(2); app.processEvents()
    assert window.session.active_source == 'follow'
    assert not window.gripper_panel.sliders['left'].isEnabled()
    assert window.follow_gripper_panel.sliders['left'].isEnabled()
    assert accepted == []  # Changing pages never moves or resets the arms.
    before = sim.q.copy()
    pad = panel.pads['left']
    window.findChild(QComboBox, 'left_follow_plane').setCurrentText('XZ')
    start = pad.point().toPoint()
    QTest.mousePress(pad, Qt.MouseButton.LeftButton, pos=start)
    QTest.keyPress(pad, Qt.Key.Key_E)
    QTest.mouseMove(pad, start + QPoint(12, 8), delay=10)
    np.testing.assert_array_equal(sim.q, before)
    window.advance(30)
    QTest.keyRelease(pad, Qt.Key.Key_E)
    QTest.mouseRelease(pad, Qt.MouseButton.LeftButton, pos=start + QPoint(12, 8))
    hand = panel.controller.hands['left'].copy()
    goal = panel.controller.targets['left'].copy()
    assert hand[0, 3] > .01 and hand[2, 3] < -.007
    assert not np.allclose(hand[:3, :3], np.eye(3))
    assert isinstance(accepted[-1], TcpTarget)
    window.advance(300)
    np.testing.assert_array_equal(panel.controller.hands['left'], hand)
    np.testing.assert_allclose(sim.targets()['left'], goal, atol=1e-12)
    assert panel.hold_metrics['left']['age'] > .5
    window.grab().save(str(directory / 'follow-hands.png'))
    # New grip rebases the target to the robot, without relocating the ball.
    here = sim.tcp_poses()['left'].copy()
    QTest.mousePress(pad, Qt.MouseButton.LeftButton, pos=pad.point().toPoint())
    np.testing.assert_array_equal(panel.controller.hands['left'], hand)
    np.testing.assert_allclose(panel.controller.targets['left'], here, atol=1e-12)
    QTest.mouseRelease(pad, Qt.MouseButton.LeftButton, pos=pad.point().toPoint())
    # Hidden legacy controls cannot overwrite Follow targets.
    count = len(accepted)
    window.left_pose.controls['x'].spin.setValue(window.left_pose.controls['x'].value()+.001)
    assert len(accepted) == count
    assert 'rejected' in window.receipt_status.text()
    tabs.setCurrentIndex(0)
    assert window.session.active_source == 'qt' and not panel.controller.active
    window.left_pose.controls['x'].spin.setValue(window.left_pose.controls['x'].value()+.001)
    assert len(accepted) == count + 1
    tabs.setCurrentIndex(2)

    # Record at a known seed and replay the exact same timed command stream.
    panel.start_recording(); window.render_timer.stop()
    assert panel.controller.recording is not None
    panel.controller.begin_grip('left')
    for i in range(50):
        panel.controller.move_hand('left', [.0001, 0., 0.], [0., .0001, 0.])
        if i == 20: window.follow_gripper_panel.sliders['left'].setValue(700)
        window.advance(2)
    panel.controller.end_grip('left'); window.advance(150)
    panel.stop()
    trace = panel.controller.trace
    trace.save(directory / 'follow-input.json')
    loaded = FollowTrace.load(directory / 'follow-input.json')
    assert len(trace.samples) == 250
    original_q = np.array([s['q'] for s in trace.samples])
    panel.play(loaded); window.render_timer.stop()
    window.advance(loaded.duration_steps + 1)
    assert panel.controller.playback is None
    replay = panel.controller.trace
    np.testing.assert_allclose(np.array([s['q'] for s in replay.samples]), original_q, atol=1e-10)
    assert replay.commands == loaded.commands
    replay.save(directory / 'follow-replayed.json')
    # Incompatible metadata and malformed commands are rejected before reset.
    incompatible = copy.deepcopy(loaded)
    incompatible.metadata['environment_sha256'] = 'wrong'
    q = sim.q.copy()
    try: panel.play(incompatible)
    except ValueError: pass
    else: raise AssertionError('incompatible recording accepted')
    np.testing.assert_array_equal(sim.q, q)
    # A different native binary is allowed for A/B, while configuration matches.
    compatible = copy.deepcopy(loaded)
    compatible.metadata['native_sha256'] = 'other-build'
    panel.play(compatible); window.render_timer.stop(); window.advance(2); panel.stop()
    # The built-in scenario feeds the same native path as direct SDK commands.
    panel.reproduce(); window.render_timer.stop()
    scenario = panel.loaded_trace
    direct = StationSimulation(settings=sim.settings, collision_profile='auto')
    direct.reset(scenario.initial_q)
    command_index = 0
    for step in range(scenario.duration_steps):
        while command_index < len(scenario.commands) and scenario.commands[command_index]['step'] == step:
            row = scenario.commands[command_index]['command']
            direct.set_target(row['tcp'], row['pose'])
            command_index += 1
        panel.runtime.advance(1)
        direct.step()
        np.testing.assert_allclose(sim.q, direct.q, atol=1e-10)
    panel.runtime.advance(1)
    assert panel.controller.playback is None
    panel.controller.trace.save(directory / 'follow-reproduction.json')
    panel.refresh(panel.runtime.samples)
    app.processEvents()
    window.grab().save(str(directory / 'follow-reproduction.png'))
    # Native fault pauses Follow; neither tab switching nor dragging resumes it.
    sim.collision_error = 'injected collision fault'
    window.advance(1)
    assert panel.controller.paused and not window.render_timer.isActive()
    tabs.setCurrentIndex(0); tabs.setCurrentIndex(2)
    assert not window.render_timer.isActive()
    assert not pad.isEnabled()
    window.grab().save(str(directory / 'follow-paused.png'))
    # External ownership is not stolen by a tab change or stale window close.
    policy = window.session.activate('policy', replace=True)
    tabs.setCurrentIndex(0); tabs.setCurrentIndex(2)
    assert window.session.active_source == 'policy'
    count = len(accepted)
    window.follow_gripper_panel.sliders['left'].setValue(800)
    assert len(accepted) == count
    window.close()
    assert window.session.active_source == 'policy'
    policy.close()
    print(json.dumps(dict(independent_ball=True, real_mouse_and_rotation=True, regrip=True,
        hold=True, exclusive_input=True, record_replay=True, gripper_recording=True, fault_pause=True)))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
