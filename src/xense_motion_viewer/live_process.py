"""Bounded, spawn-based live display. This module does not import Rerun."""
import logging
import multiprocessing as mp
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .live_panels import ImageStream, TaskEvent, TaskSnapshot
from .streams import DisplayStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessDisplayConfig:
    display_hz: float = 30.0
    image_hz: float = 10.0
    image_max_edge: int = 640
    jpeg_quality: int = 85
    stale_after: float = 1.0
    capacity: int = 8
    max_image_bytes: int = 8 * 1024 * 1024
    spawn_viewer: bool = True
    viewer_port: int = 9877
    save_path: str | None = None
    startup_timeout: float = 30.0

    def __post_init__(self):
        for key in ('display_hz', 'image_hz', 'stale_after', 'startup_timeout'):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('image_max_edge', 'capacity', 'max_image_bytes'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.jpeg_quality) is not int or not 1 <= self.jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be between 1 and 100')
        if type(self.viewer_port) is not int or not 1 <= self.viewer_port <= 65535:
            raise ValueError('Invalid viewer_port')
        if not isinstance(self.spawn_viewer, bool):
            raise TypeError('spawn_viewer must be a boolean')
        if not self.spawn_viewer and not self.save_path:
            raise ValueError('Live display needs a Viewer or a recording file')


class _Channel:
    """Fixed byte capacity; neither endpoint waits for the other endpoint's lock."""
    def __init__(self, ctx, capacity, size):
        self.capacity, self.size = capacity, size
        self.buffer = ctx.RawArray('B', capacity * size)
        self.lengths = ctx.RawArray('I', capacity)
        self.count = ctx.RawValue('I', 0)
        self.lock = ctx.Lock()

    def put(self, payload, *, latest=False):
        if len(payload) > self.size:
            raise ValueError(f'Display packet exceeds {self.size} bytes')
        if not self.lock.acquire(False):
            return False
        try:
            if self.count.value == self.capacity and not latest:
                return False
            slot = 0 if latest else self.count.value
            offset = slot * self.size
            memoryview(self.buffer).cast('B')[offset:offset + len(payload)] = payload
            self.lengths[slot] = len(payload)
            self.count.value = 1 if latest else slot + 1
            return True
        finally:
            self.lock.release()

    def take(self):
        if not self.lock.acquire(False):
            return ()
        try:
            view = memoryview(self.buffer).cast('B')
            result = tuple(bytes(view[i * self.size:i * self.size + self.lengths[i]])
                           for i in range(self.count.value))
            self.count.value = 0
            return result
        finally:
            self.lock.release()


def _description_wire(description):
    result = {f.name: getattr(description, f.name) for f in fields(description)}
    for key in ('joint_defaults', 'tool_frames', 'package_roots'):
        result[key] = dict(result[key])
    return result


def _frame_wire(frame):
    state = None
    if frame.state is not None:
        state = {f.name: getattr(frame.state, f.name) for f in fields(frame.state)}
        for key in ('joint_positions', 'tcp_poses', 'targets'):
            state[key] = dict(state[key])
    return (frame.time, {k: (v.step_ids, v.poses) for k, v in frame.futures.items()},
            state, frame.sample_id, frame.reset, frame.status)


def _restore_frame(data):
    from .contracts import RobotState
    from .live import FutureTrajectory, LiveFrame
    at, futures, state, sample_id, reset, status = data
    return LiveFrame(at, {k: FutureTrajectory(*v) for k, v in futures.items()},
                     None if state is None else RobotState(**state), sample_id, reset, status)


def _display_main(description, config, streams, channels, stop, ready, counters, clock_origin):
    recording = None
    try:
        import os

        import rerun as rr

        from .contracts import RobotDescription
        from .live import LiveRerunRenderer
        from .live_panels import LivePanels

        rr.set_strict_mode(True)
        recording = rr.RecordingStream('xense-motion-viewer-live',
            batcher_config=rr.ChunkBatcherConfig(flush_tick=1 / config.display_hz))
        sinks = []
        if config.save_path:
            path = Path(config.save_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            sinks.append(rr.FileSink(path))
        if config.spawn_viewer:
            recording.spawn(port=config.viewer_port, connect=False, memory_limit='512MiB')
            sinks.append(rr.GrpcSink(f'rerun+http://127.0.0.1:{config.viewer_port}/proxy'))
        recording.set_sinks(*sinks)
        motion = LiveRerunRenderer(recording, RobotDescription(**description))
        panels = LivePanels(recording, motion, streams, max_edge=config.image_max_edge,
                            jpeg_quality=config.jpeg_quality, stale_after=config.stale_after)
        ready.send((True, os.getpid()))
        ready.close()
        last_flush = time.monotonic()
        while True:
            packets = []
            for channel in channels.values():
                packets.extend(pickle.loads(raw) for raw in channel.take())
            if packets:
                generation = max(packet[1] for packet in packets)
                at = min(packet[2] for packet in packets if packet[1] == generation)
                panels.advance(generation, at)
                frames = []
                for kind, epoch, timestamp, data in packets:
                    if epoch != panels.generation:
                        continue
                    try:
                        if kind == 'motion':
                            frames.append(_restore_frame(data))
                        elif kind == 'task':
                            panels.task(data)
                        else:
                            key, rgb = data
                            if time.monotonic() - clock_origin - timestamp <= config.stale_after:
                                panels.image(key, timestamp, rgb)
                        if kind != 'motion':
                            counters[0] += 1
                    except Exception:
                        logger.exception("Live display packet failed")
                        counters[1] += 1
                if frames:
                    try:
                        panels.motion_frames(frames)
                        counters[0] += len(frames)
                    except Exception:
                        logger.exception("Live display packet failed")
                        counters[1] += 1
            panels.expire(time.monotonic() - clock_origin)
            # A stalled sink must not accumulate an unbounded SDK send backlog.
            if time.monotonic() - last_flush >= 1:
                recording.flush(timeout_sec=.5)
                last_flush = time.monotonic()
            if stop.value and not packets:
                break
            if not stop.value:
                time.sleep(1 / config.display_hz)
        recording.flush(timeout_sec=2)
    except Exception as exc:
        logger.exception("Live display process stopped")
        counters[1] += 1
        try:
            ready.send((False, f'{type(exc).__name__}: {exc}'))
        except (BrokenPipeError, OSError):
            pass
        # After a sink failure, disconnect and Rerun's atexit can wait forever.
        # Exit only this isolated worker; its parent observes the failed process.
        os._exit(1)
    finally:
        ready.close()
        if recording is not None:
            recording.disconnect()


class LiveDisplayProcess:
    """All Rerun work runs in a spawned child; publication never waits for IPC.

    publish_images borrows HWC RGB arrays: producers must not modify or reuse
    their storage after publication. The feeder takes a stable copy off the
    observation thread. Current Elite camera drivers replace their RGB arrays.
    Motion frames are immutable. Each generation clears previous display state.
    """
    def __init__(self, description, *, config=None, streams=(), clock_origin=None):
        self.config = config or ProcessDisplayConfig()
        self.streams = tuple(streams)
        streams = self.streams
        if len({s.key for s in streams}) != len(streams):
            raise ValueError('Image stream keys must be unique')
        self.clock_origin = time.monotonic() if clock_origin is None else clock_origin
        self._lock, self._wake = threading.Lock(), threading.Event()
        self._motion, self._images = deque(), {}
        self._task = None
        self._task_revision = (-1, -1., -1)
        self._in_flight = self._discarded = 0
        self._closing_deadline = None
        self._next_image = 0.0
        self._closed = False
        self._accepted = self._dropped = self._rejected = self._errors = self._coalesced = 0
        self._last_error = None
        self._ctx = mp.get_context('spawn')
        self._channels = {'motion': _Channel(self._ctx, self.config.capacity, 2 * 1024 * 1024),
                          'task': _Channel(self._ctx, 1, 256 * 1024)}
        self._channels.update({s.key: _Channel(self._ctx, 1, self.config.max_image_bytes) for s in streams})
        if any(s.key in ('motion', 'task') for s in streams):
            raise ValueError('Reserved image stream name')
        # A terminated child can leave an Event's condition lock acquired.
        self._stop = self._ctx.RawValue('b', False)
        self._counters = self._ctx.RawArray('Q', 2)
        reader, writer = self._ctx.Pipe(duplex=False)
        self._process = self._ctx.Process(target=_display_main, name='xmv-display', daemon=True,
            args=(_description_wire(description), self.config, self.streams, self._channels,
                  self._stop, writer, self._counters, self.clock_origin))
        self._process.start()
        writer.close()
        try:
            if not reader.poll(self.config.startup_timeout):
                raise TimeoutError('XMV display process startup timed out')
            success, result = reader.recv()
            if not success:
                raise RuntimeError(f'XMV display process startup failed: {result}')
        except BaseException:
            self._process.terminate()
            self._process.join(2)
            raise
        finally:
            reader.close()
        self._feeder = threading.Thread(target=self._transfer, name='xmv-data-transfer', daemon=True)
        self._feeder.start()

    @property
    def pid(self):
        return self._process.pid

    def _enter(self):
        if self._closed or not self._process.is_alive():
            self._rejected += 1
            return False
        if not self._lock.acquire(False):
            self._dropped += 1
            return False
        return True

    def publish(self, frame, *, generation=0):
        """Reject without waiting; callers must retry reset/absence boundaries."""
        if not self._enter():
            return False
        try:
            if len(self._motion) >= self.config.capacity:
                self._dropped += 1
                return False
            self._motion.append((generation, frame))
            self._accepted += 1
            self._wake.set()
            return True
        finally:
            self._lock.release()

    def publish_images(self, observed_at, images, *, generation=0):
        if observed_at < self._next_image or not self._enter():
            return False
        try:
            self._next_image = observed_at + 1 / self.config.image_hz
            for stream in self.streams:
                rgb = images.get(stream.key)
                if rgb is not None:
                    self._coalesced += int(stream.key in self._images)
                    self._images[stream.key] = (generation, observed_at, rgb)
                    self._accepted += 1
            self._wake.set()
            return True
        finally:
            self._lock.release()

    def update_task(self, snapshot):
        if not self._enter():
            return False
        try:
            revision = (snapshot.generation, snapshot.time,
                        snapshot.events[-1].sequence if snapshot.events else -1)
            if revision <= self._task_revision:
                return True
            self._task_revision = revision
            self._task = snapshot
            self._accepted += 1
            self._wake.set()
            return True
        finally:
            self._lock.release()

    def _send(self, key, message, *, latest=False):
        try:
            if not self._channels[key].put(pickle.dumps(message, protocol=5), latest=latest):
                self._dropped += 1
                return False
            return True
        except Exception as exc:
            logger.exception("Live display transfer failed")
            self._errors += 1
            self._last_error = f'{type(exc).__name__}: {exc}'
            return None  # Invalid packets cannot be retried unchanged.

    def _transfer(self):
        pending = deque()
        force_reset = False
        while True:
            self._wake.wait(.01 if pending else .1)
            with self._lock:
                frames = () if pending else tuple(self._motion)
                if not pending:
                    self._motion.clear()
                images, task = self._images, self._task
                self._images, self._task = {}, None
                self._wake.clear()
            for generation, frame in frames:
                try:
                    data = list(_frame_wire(frame))
                    data[4] = data[4] or force_reset
                    payload = pickle.dumps(('motion', generation, frame.time, tuple(data)), protocol=5)
                    if len(payload) > self._channels['motion'].size:
                        raise ValueError('Motion packet exceeds display channel capacity')
                    pending.append(payload)
                    force_reset = False
                except Exception as exc:
                    logger.exception('Live motion transfer failed')
                    self._errors += 1
                    self._last_error = str(exc)
                    self._discarded += 1
                    force_reset = True
            while pending and self._channels['motion'].put(pending[0]):
                pending.popleft()
            self._in_flight = len(pending)
            if task is not None:
                sent = self._send('task', ('task', task.generation, task.time, task), latest=True)
                if sent is False:
                    with self._lock:
                        if self._task is None:
                            self._task = task
            for key, (generation, at, rgb) in images.items():
                self._send(key, ('image', generation, at, (key, rgb)), latest=True)
            with self._lock:
                drained = not pending and not self._motion and not self._images and self._task is None
                expired = self._closing_deadline is not None and time.monotonic() >= self._closing_deadline
                if not self._process.is_alive() or expired:
                    self._discarded += len(pending) + len(self._motion) + len(self._images) + int(self._task is not None)
                    self._motion.clear()
                    self._images.clear()
                    self._task = None
                    self._in_flight = 0
                    return
                if self._closed and drained:
                    return

    @property
    def stats(self):
        failed = self._process.exitcode not in (None, 0)
        return DisplayStats(self._accepted, int(self._counters[0]), self._dropped, self._discarded,
                            self._rejected, self._errors + int(self._counters[1]) + int(failed and not self._counters[1]),
                            len(self._motion) + len(self._images) + self._in_flight + int(self._task is not None), self._closed,
                            self._process.is_alive() or self._feeder.is_alive(),
                            self._last_error or (f'Display process exited: {self._process.exitcode}' if failed else None),
                            self._coalesced)

    def close(self, *, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closed = True
            self._closing_deadline = deadline
            self._wake.set()
        self._feeder.join(max(0, deadline - time.monotonic()))
        self._stop.value = True
        self._process.join(max(0, deadline - time.monotonic()))
        graceful = not self._process.is_alive() and not self._feeder.is_alive()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(1)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1)
        self._wake.set()
        self._feeder.join(1)
        return graceful


__all__ = ['ImageStream', 'LiveDisplayProcess', 'ProcessDisplayConfig', 'TaskEvent', 'TaskSnapshot']
