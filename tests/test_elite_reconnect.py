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


@pytest.mark.parametrize('display_action', ['open_viewer', 'follow_live'])
def test_open_request_survives_state_coalescing_and_stays_on_display_owner(display_action):
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
    worker = _LatestDisplay(consume, lambda: None, hz=1000, open_callback=open_viewer, follow_callback=open_viewer)
    worker.publish(RobotState(0, {'q': 0}), False)
    assert entered.wait(1)
    worker.publish(RobotState(1, {'q': 1}), False, **{display_action: True})
    worker.publish(RobotState(2, {'q': 2}), True)
    release.set(); assert worker.close(timeout=1)
    expected = [('open', 2., True), ('state', 2., True)] if display_action == 'open_viewer' else [
        ('state', 2., True), ('open', 2., True)]
    assert calls == [('state', 0., False), *expected]
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


def test_gripper_edit_resumes_current_recording_without_changing_camera_or_blueprint(monkeypatch):
    from rerun.experimental import ViewerClient
    calls = []
    client = SimpleNamespace(
        viewer_state=lambda: SimpleNamespace(active_recording='Recording:app:abc'),
        set_time=lambda *args, **kwargs: calls.append(('time', args, kwargs)))
    monkeypatch.setattr(ViewerClient, 'connect', lambda url: client)
    view = EliteRerunView.__new__(EliteRerunView)
    view.port, view._last_follow = 9876, 0.
    view.renderer = SimpleNamespace(timeline='sandbox_elapsed', recording=SimpleNamespace(
        get_recording_id=lambda: 'abc', flush=lambda **kwargs: calls.append(('flush', kwargs))))
    assert view._follow_live(RobotState(12.5, {'q': 0}), False)
    assert calls == [('flush', {'timeout_sec': .5}), ('time', ('sandbox_elapsed',),
                     dict(duration=12.5, play=True, recording='Recording:app:abc'))]
    # A drag's repeated valueChanged events do not cause per-frame RPCs.
    assert not view._follow_live(RobotState(12.51, {'q': 0}), False)
    assert len(calls) == 2
