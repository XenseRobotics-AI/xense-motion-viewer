"""Narrow Viewer lifecycle regressions; no native scheduling changes."""
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from robot_motion_workbench import RobotState
from robot_motion_workbench.adapters.elite import EliteRerunView, _LatestDisplay, _open_recording_viewer


def test_finish_always_disconnects_after_failed_flush():
    calls = []
    def flush(**kwargs):
        calls.append(('flush', kwargs['timeout_sec']))
        raise RuntimeError('flush timed out')
    view = EliteRerunView.__new__(EliteRerunView)
    view.renderer = SimpleNamespace(recording=SimpleNamespace(flush=flush, disconnect=lambda: calls.append(('disconnect',))))
    with pytest.raises(RuntimeError, match='flush timed out'):
        view._finish()
    assert calls == [('flush', .5), ('disconnect',)]
    assert view.renderer.recording._rmw_server is None


def test_open_request_survives_state_coalescing_and_stays_on_display_owner():
    entered, release = Event(), Event()
    calls = []
    owner = get_ident()
    def consume(state, visible):
        assert get_ident() != owner
        calls.append(('state', state.time, visible))
        if state.time == 0:
            entered.set(); assert release.wait(2)
    def open_viewer(state, visible):
        assert get_ident() != owner
        calls.append(('open', state.time, visible))
    worker = _LatestDisplay(consume, lambda: None, hz=1000, open_callback=open_viewer)
    worker.publish(RobotState(0, {'q': 0}), False)
    assert entered.wait(1)
    worker.publish(RobotState(1, {'q': 1}), False, open_viewer=True)
    worker.publish(RobotState(2, {'q': 2}), True)
    release.set(); assert worker.close(timeout=1)
    assert calls == [('state', 0., False), ('open', 2., True), ('state', 2., True)]
    assert not worker.stats.alive and worker.stats.errors == 0


def test_already_connected_open_keeps_current_blueprint_camera_and_sinks(monkeypatch):
    import rerun as rr
    from rerun.experimental import ViewerClient
    calls = []
    def unexpected(*args):
        raise AssertionError('Already active recording must not be reopened or replayed')
    client = SimpleNamespace(viewer_state=lambda: SimpleNamespace(active_recording='Recording:app:abc'), open_url=unexpected)
    monkeypatch.setattr(rr, 'spawn', lambda **kwargs: calls.append('ensure-viewer'))
    monkeypatch.setattr(ViewerClient, 'connect', lambda url: client)
    recording = SimpleNamespace(_rmw_server=SimpleNamespace(uri='unused'), get_recording_id=lambda: 'abc')
    assert _open_recording_viewer(recording, 12345, prepare=unexpected) is False
    assert calls == ['ensure-viewer']
