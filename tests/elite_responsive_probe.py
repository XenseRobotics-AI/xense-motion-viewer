"""Real Qt signals/native planner while the display worker is deliberately blocked."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from threading import Event, get_ident
import time

import numpy as np
import rerun as rr
from libpyelite.sim import StationSimulation
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton
from xense_motion_viewer.adapters.elite import EliteRerunView, make_qt_window

root = Path(sys.argv[1]); root.mkdir(exist_ok=True)
sim = StationSimulation(collision_profile='auto')
recording = rr.RecordingStream('responsive'); recording.save(root/'responsive.rrd')
view = EliteRerunView(sim, recording)
owner = get_ident(); snapshot = view.adapter.snapshot
def checked_snapshot():
    assert get_ident() == owner
    return snapshot()
view.adapter.snapshot = checked_snapshot
entered, release = Event(), Event()
render = view.renderer.update
render_threads = []
def blocked(*args, **kwargs):
    render_threads.append(get_ident())
    if not release.is_set():
        entered.set()
        assert release.wait(3)
    return render(*args, **kwargs)
view.renderer.update = blocked
app, window = make_qt_window(sim, view)
window.show()
toggle = next(b for b in window.findChildren(QPushButton) if b.text() == '显示碰撞包络')
joint = window.left_joints.controls[0]; initial_joint = joint.slider.value()
pose = window.left_pose.controls['x']; initial_pose = pose.slider.value()
before = sim.q.copy(); ticks = []; callbacks = []; errors = []
start = time.perf_counter()
def tick():
    try:
        now = time.perf_counter(); elapsed = now-start; ticks.append(now)
        t = time.perf_counter()
        if elapsed < .25:
            joint.slider.setValue(initial_joint + (len(ticks) % 3))
        else:
            pose.slider.setValue(initial_pose + (len(ticks) % 2))
        toggle.setChecked(len(ticks) % 2 == 0)
        callbacks.append(time.perf_counter()-t)
        if elapsed > .6:
            assert entered.is_set()
            assert not np.array_equal(sim.q, before), 'Native motion stopped with blocked display'
            window.render_timer.stop()
            before_reset_time = view._last_state.time
            window._reset_home(); window.render_timer.stop(); window.advance(4)
            assert view._last_state.time >= before_reset_time, 'Display time moved backwards after reset'
            toggle.setChecked(True)
            timer.stop(); release.set(); app.quit()
    except Exception as exc:
        errors.append(repr(exc)); release.set(); timer.stop(); app.quit()
timer = QTimer(); timer.setInterval(10); timer.timeout.connect(tick); timer.start()
app.exec()
window.close()
assert not errors, errors
assert len(ticks) >= 35
assert max(callbacks) < .05
assert view.stats.pending == 0 and not view.stats.alive
assert view.stats.rendered_time == view.stats.latest_time
assert view.renderer.collision_visible
assert view.stats.coalesced > 10 and view.stats.errors == 0
assert render_threads and all(t != owner for t in render_threads)
# Reopen with a fresh recording/view: no old thread or clock epoch is reused.
rec2 = rr.RecordingStream('reopened'); rec2.save(root/'reopened.rrd')
view2 = EliteRerunView(sim, rec2)
_, window2 = make_qt_window(sim, view2); window2.render_timer.stop(); window2.advance(4); window2.close()
assert not view2.stats.alive and view2.stats.rendered_time == view2.stats.latest_time
result = dict(gui_ticks=len(ticks),max_callback_ms=max(callbacks)*1000,
    max_event_gap_ms=float(np.max(np.diff(ticks)))*1000,display=asdict(view.stats),
    native_motion_while_blocked=True,reset_monotonic=True,new_window_closed=True,sdk_on_owner=True)
(root/'responsive.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
