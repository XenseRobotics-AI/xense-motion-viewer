import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from rrd_reader import RrdReader, clear_times

from xense_motion_viewer import RobotDescription, RobotState
from xense_motion_viewer.live import LiveFrame
from xense_motion_viewer.live_process import (
    ImageStream,
    LiveDisplayProcess,
    ProcessDisplayConfig,
    TaskEvent,
    TaskSnapshot,
)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Display did not make expected progress')
        time.sleep(.01)


def values(chunks, path, component):
    return [value for chunk in chunks if chunk.entity_path == '/' + path
            for batch in [chunk.to_record_batch()] if component in batch.schema.names
            for value in batch.column(component).to_pylist() if value is not None]


def test_child_records_images_tasks_and_independent_motion(robot_urdf, names, tmp_path):
    path = tmp_path / 'process.rrd'
    image_keys = ('head', 'left_wrist', 'right_wrist', 'left_tactile_left',
                  'right_tactile_left', 'left_tactile_right', 'right_tactile_right')
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(path), stale_after=60, capacity=32,
                                    image_max_edge=16),
        streams=tuple(ImageStream(key, key.replace('_', ' ').title()) for key in image_keys))
    assert display.pid != os.getpid()
    red = np.full((30, 50, 3), [250, 0, 0], dtype=np.uint8)
    copy = red.copy()
    state = RobotState(.1, dict.fromkeys(names, 0.), tcp_poses={'left': np.eye(4)})
    try:
        assert display.update_task(TaskSnapshot(.1, 1, 1, 'Pick up the cup', 'policy default', 'Running',
                                              (TaskEvent(1, .1, 'Episode 1 started'),)))
        assert display.publish_images(.1, dict.fromkeys(image_keys, red), generation=1)
        assert display.publish(LiveFrame(.1, state=state, sample_id='a'), generation=1)
        wait_for(lambda: display.stats.consumed >= 9)
        assert display.publish(LiveFrame(.2, state=None), generation=1)
        assert display.publish(LiveFrame(.3, state=state, sample_id='b'), generation=1)
        wait_for(lambda: display.stats.consumed >= 11)
    finally:
        assert display.close()
    assert display.stats.errors == 0
    assert not display.stats.alive
    assert display.close()
    assert not display.publish(LiveFrame(1))
    np.testing.assert_array_equal(red, copy)
    chunks = list(RrdReader(path).stream())
    blob = values(chunks, 'live/images/head', 'EncodedImage:blob')[0][0]
    picture = Image.open(BytesIO(bytes(blob)))
    assert picture.size == (16, 10)
    color = np.asarray(picture).mean(axis=(0, 1))
    assert color[0] > 240 and max(color[1:]) < 10
    for key in image_keys:
        assert values(chunks, f'live/images/{key}', 'EncodedImage:blob')
    text = str(values(chunks, 'live/task/current', 'TextDocument:text'))
    assert 'Pick up the cup' in text and 'Running' in text
    assert 'Not connected' in str(values(chunks, 'live/task/subtask', 'TextDocument:text'))
    assert 'Episode 1 started' in str(values(chunks, 'live/task/events', 'TextLog:text'))
    assert values(chunks, 'live/task/timeline/task', 'Scalars:scalars')
    assert values(chunks, 'live/task/timeline/task', 'SeriesLines:colors')
    assert clear_times(chunks, 'world/live/robot/measured/left/current', 'host_elapsed')
    assert values(chunks, 'world/live/robot/measured/left/history/1', 'LineStrips3D:strips')
    assert str(display.pid) in str(values(chunks, 'live/display', 'process_id'))


def test_generation_clears_old_images_and_rejects_late_old_frames(robot_urdf, names, tmp_path):
    path = tmp_path / 'epochs.rrd'
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(path), stale_after=60),
        streams=(ImageStream('camera', 'Camera'),))
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        display.publish_images(.1, {'camera': pixels}, generation=1)
        wait_for(lambda: display.stats.consumed >= 1)
        display.update_task(TaskSnapshot(.2, generation=2, episode=2))
        wait_for(lambda: display.stats.consumed >= 2)
        display.publish_images(.4, {'camera': pixels}, generation=1)
        display.publish(LiveFrame(.5), generation=2)
        wait_for(lambda: display.stats.consumed >= 3)
    finally:
        assert display.close()
    assert display.stats.errors == 0
    chunks = list(RrdReader(path).stream())
    assert len(values(chunks, 'live/images/camera', 'EncodedImage:blob')) == 1
    assert clear_times(chunks, 'live/images/camera', 'host_elapsed')


def test_image_publication_does_not_copy_and_busy_channels_do_not_block(robot_urdf, names, tmp_path):
    class BorrowOnly(np.ndarray):
        def __reduce_ex__(self, protocol):
            raise AssertionError('Only the background feeder may serialize this test image')

    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(tmp_path / 'busy.rrd')),
        streams=(ImageStream('camera', 'Camera'),))
    image = np.zeros((4, 4, 3), dtype=np.uint8).view(BorrowOnly)
    try:
        # Serialization failures are contained by the background transfer worker.
        assert display.publish_images(.1, {'camera': image})
        wait_for(lambda: display.stats.errors == 1)
        display._lock.acquire()
        try:
            assert not display.publish_images(.3, {'camera': image})
            assert not display.publish(LiveFrame(.3))
        finally:
            display._lock.release()
        display._process.terminate()
        display._process.join(3)
        assert not display.publish(LiveFrame(.4))
        assert not display.publish_images(.5, {'camera': image})
    finally:
        display.close()
    assert not display.stats.alive


def test_no_sink_rejected_and_startup_failure_does_not_leave_worker(robot_urdf, names, tmp_path):
    with pytest.raises(ValueError, match='Viewer or a recording'):
        ProcessDisplayConfig(spawn_viewer=False)
    blocked = tmp_path / 'file'
    blocked.write_text('not a directory')
    with pytest.raises(RuntimeError, match='startup failed'):
        LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
            config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(blocked / 'bad.rrd')))


@pytest.mark.parametrize('reset', [False, True])
def test_motion_boundaries_survive_a_busy_ipc_channel(robot_urdf, names, tmp_path, reset):
    path = tmp_path / 'busy-motion.rrd'
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(path)))
    def state(x):
        pose = np.eye(4)
        pose[0, 3] = x
        return RobotState(.1, dict.fromkeys(names, 0.), tcp_poses={'tcp': pose})
    try:
        display.publish(LiveFrame(.1, state=state(1), sample_id='a'), generation=1)
        wait_for(lambda: display.stats.consumed >= 1)
        display._channels['motion'].lock.acquire()
        try:
            assert display.publish(LiveFrame(.2, reset=reset), generation=1)
            wait_for(lambda: display._in_flight == 1)
            assert display.publish(LiveFrame(.3, state=state(2), sample_id='b'), generation=1)
        finally:
            display._channels['motion'].lock.release()
        wait_for(lambda: display.stats.consumed >= 3)
    finally:
        assert display.close()
    chunks = list(RrdReader(path).stream())
    history = values(chunks, 'world/live/robot/measured/tcp/history/0', 'LineStrips3D:strips')
    assert all(len(row[0]) == 1 for row in history)
    assert display.stats.errors == 0


def test_task_arrival_does_not_change_motion_time_or_accept_older_task(robot_urdf, names, tmp_path):
    path = tmp_path / 'independent-time.rrd'
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(path)))
    try:
        newer = TaskSnapshot(.2, generation=1, status='NEW', events=(TaskEvent(2, .2, 'NEW_EVENT'),))
        older = TaskSnapshot(.1, generation=1, status='OLD', events=(TaskEvent(1, .1, 'OLD_EVENT'),))
        display.update_task(newer)
        wait_for(lambda: display.stats.consumed >= 1)
        display.update_task(older)
        display.publish(LiveFrame(.1, state=RobotState(.1, dict.fromkeys(names, 0.)), sample_id='a'), generation=1)
        wait_for(lambda: display.stats.consumed >= 2)
    finally:
        assert display.close()
    assert display.stats.errors == 0
    chunks = list(RrdReader(path).stream())
    tasks = str(values(chunks, 'live/task/current', 'TextDocument:text'))
    assert 'NEW' in tasks and 'OLD' not in tasks
    assert any(c.entity_path.startswith('/world/robots/robot') and not c.is_static for c in chunks)


def test_rolling_task_window_only_marks_events_the_consumer_missed(robot_urdf, names, tmp_path):
    path = tmp_path / 'events.rrd'
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(path)))
    try:
        events = tuple(TaskEvent(i, i / 1000, f'event-{i}') for i in range(1, 257))
        display.update_task(TaskSnapshot(1, events=events))
        wait_for(lambda: display.stats.consumed >= 1)
        events = (*events[1:], TaskEvent(257, .257, 'event-257'))
        display.update_task(TaskSnapshot(2, events=events, dropped_events=1))
        wait_for(lambda: display.stats.consumed >= 2)
        display.update_task(TaskSnapshot(3, events=(TaskEvent(260, 3, 'event-260'),), dropped_events=259))
        wait_for(lambda: display.stats.consumed >= 3)
    finally:
        assert display.close()
    logs = values(list(RrdReader(path).stream()), 'live/task/events', 'TextLog:text')
    gaps = [text for row in logs for text in row if 'event gap' in text]
    assert gaps == ['Runtime event gap: 2 missing']


def test_host_never_imports_rerun_even_when_display_is_active(robot_urdf, names, tmp_path):
    script = tmp_path / 'host.py'
    script.write_text(f'''
import sys
from xense_motion_viewer import RobotDescription
from xense_motion_viewer.live_process import LiveDisplayProcess, ProcessDisplayConfig

if __name__ == '__main__':
    assert 'rerun' not in sys.modules
    display = LiveDisplayProcess(RobotDescription('robot', {str(robot_urdf)!r}, {names!r}),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path={str(tmp_path / 'host.rrd')!r}))
    try:
        assert 'rerun' not in sys.modules
    finally:
        assert display.close()
''')
    subprocess.run([sys.executable, str(script)], check=True, timeout=20)


def test_close_with_busy_motion_transport_stops_and_can_be_repeated(robot_urdf, names, tmp_path):
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(tmp_path / 'close.rrd')))
    display._channels['motion'].lock.acquire()
    try:
        assert display.publish(LiveFrame(.1, reset=True))
        wait_for(lambda: display._in_flight == 1)
        display.close(timeout=.05)
    finally:
        display._channels['motion'].lock.release()
    assert display.close()
    assert not display.stats.alive
    assert display.stats.discarded_close >= 1


def test_close_drains_image_published_during_previous_transfer(robot_urdf, names, tmp_path, monkeypatch):
    path = tmp_path / 'close-images.rrd'
    display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
        config=ProcessDisplayConfig(spawn_viewer=False, save_path=str(path), stale_after=60),
        streams=(ImageStream('camera', 'Camera'),))
    entered, release = threading.Event(), threading.Event()
    real_send = display._send

    def paused_send(key, *args, **kwargs):
        sent = real_send(key, *args, **kwargs)
        if key == 'camera' and not entered.is_set():
            # Consume the first image so the latest-only IPC slot cannot coalesce it.
            wait_for(lambda: display.stats.consumed == 1)
            entered.set()
            assert release.wait(5)
        return sent

    monkeypatch.setattr(display, '_send', paused_send)
    closed = []
    closer = threading.Thread(target=lambda: closed.append(display.close(timeout=5)))
    try:
        assert display.publish_images(.1, {'camera': np.zeros((4, 4, 3), dtype=np.uint8)}, generation=1)
        assert entered.wait(5)
        assert display.publish_images(.3, {'camera': np.full((4, 4, 3), 255, dtype=np.uint8)}, generation=1)
        closer.start()
        wait_for(lambda: display.stats.closed)
        release.set()
        closer.join(6)
        assert not closer.is_alive()
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(6)
        display.close()

    assert closed == [True]
    stats = display.stats
    assert stats.accepted == stats.consumed == 2
    assert stats.pending == 0
    assert stats.dropped_full == stats.discarded_close == stats.errors == stats.coalesced == 0
    assert not stats.alive
    chunks = list(RrdReader(path).stream())
    rows = [(at, image) for chunk in chunks if chunk.entity_path == '/live/images/camera'
            for batch in [chunk.to_record_batch()] if 'EncodedImage:blob' in batch.schema.names
            for at, image in zip(batch.column('host_elapsed').cast('int64').to_pylist(),
                                 batch.column('EncodedImage:blob').to_pylist()) if image]
    assert [at for at, _ in rows] == [100000000, 300000000]
    for (_, image), color in zip(rows, (0, 255), strict=True):
        picture = Image.open(BytesIO(bytes(image[0])))
        assert np.all(np.asarray(picture) == color)


def _display_without_window(*args):
    import rerun as rr

    from xense_motion_viewer.live_process import _display_main

    rr.RecordingStream.spawn = lambda *args, **kwargs: None
    _display_main(*args)


def test_failed_sink_stops_child_and_feeder_without_explicit_close(robot_urdf, names, monkeypatch):
    import xense_motion_viewer.live_process as module

    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    listener.settimeout(.1)
    release = threading.Event()

    def unresponsive_peer():
        while not release.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                release.wait(10)
            return

    peer = threading.Thread(target=unresponsive_peer, daemon=True)
    peer.start()
    monkeypatch.setattr(module, '_display_main', _display_without_window)
    display = None
    try:
        display = LiveDisplayProcess(RobotDescription('robot', robot_urdf, names),
            config=ProcessDisplayConfig(viewer_port=listener.getsockname()[1], startup_timeout=3))
        wait_for(lambda: display.stats.errors > 0, timeout=5)
        wait_for(lambda: not display.stats.alive, timeout=3)
        assert not display.publish(LiveFrame(5))
    finally:
        if display is not None:
            display.close(timeout=.2)
        release.set()
        peer.join(2)
        listener.close()
