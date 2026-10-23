"""Exercise the retained production sliders through the common command API."""
import json
from pathlib import Path
import sys

import numpy as np
import rerun as rr
from libpyelite.sim import StationSimulation
from PySide6.QtWidgets import QTabWidget

from robot_motion_workbench import JointTarget, TcpTarget
from robot_motion_workbench.adapters.elite import EliteRerunView, make_qt_window


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
    policy = window.session.activate('policy', replace=True)
    pose = sim.tcp_poses()['left'].copy(); pose[0, 3] += .001
    policy.submit(TcpTarget('left', pose))
    assert len(accepted) == 3
    control.slider.setValue(control.slider.value() + 1)
    assert len(accepted) == 3
    assert 'rejected' in window.receipt_status.text()
    snapshot = sim.q.copy()
    window._reset_home()
    np.testing.assert_array_equal(sim.q, snapshot)  # old Qt lease cannot reset policy state
    window.advance(4)
    assert len(accepted) == 3
    window.close()
    assert window.session.active_source == 'policy'  # stale Qt close cannot release policy
    view.renderer.flush(); recording.disconnect()
    result = dict(original_sliders=True, joint_and_tcp_via_common_api=True,
                  observation_refresh_does_not_submit=True, readable_diagnostics=True,
                  ownership_rejects_stale_qt=True, submitted=3)
    (directory / 'elite-qt-results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
