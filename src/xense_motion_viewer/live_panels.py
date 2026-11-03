"""Image and task presentation, used only by the isolated display consumer."""
import os
from dataclasses import dataclass, replace

import numpy as np

from .contracts import name
from .policy import elapsed
from .rerun_renderer import segment


@dataclass(frozen=True)
class ImageStream:
    key: str
    label: str

    def __post_init__(self):
        name(self.key)
        name(self.label)


@dataclass(frozen=True)
class TaskEvent:
    sequence: int
    time: float
    text: str

    def __post_init__(self):
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError('Task event sequence must be a positive integer')
        elapsed(self.time)
        if not isinstance(self.text, str):
            raise TypeError('Task event text must be a string')


@dataclass(frozen=True)
class TaskSnapshot:
    time: float
    generation: int = 0
    episode: int = 0
    instruction: str | None = None
    source: str = ''
    status: str = 'Waiting for runtime'
    events: tuple[TaskEvent, ...] = ()
    dropped_events: int = 0

    def __post_init__(self):
        elapsed(self.time)
        if any(type(value) is not int or value < 0
               for value in (self.generation, self.episode, self.dropped_events)):
            raise ValueError('Task counters must be nonnegative integers')
        object.__setattr__(self, 'events', tuple(self.events))
        if len(self.events) > 256:
            raise ValueError('Task history is limited to 256 events')
        if self.instruction is not None and not isinstance(self.instruction, str):
            raise TypeError('Task instruction must be text or None')
        previous = -1
        for event in self.events:
            if not isinstance(event, TaskEvent):
                raise TypeError('Task history must contain TaskEvent values')
            elapsed(event.time)
            if event.time > self.time:
                raise ValueError('Task events cannot occur after their snapshot')
            if event.sequence <= previous:
                raise ValueError('Task event sequences must increase')
            previous = event.sequence


class LivePanels:
    def __init__(self, recording, motion, streams, *, max_edge, jpeg_quality, stale_after):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr, self.recording, self.motion = rr, recording, motion
        self.streams = {stream.key: stream for stream in streams}
        self.max_edge, self.jpeg_quality, self.stale_after = max_edge, jpeg_quality, stale_after
        self._image_times = {}
        self._last_event = -1
        self._motion_reset_pending = False
        self.generation = -1
        world = rrb.Spatial3DView(origin='world', name='Robot / reference / measured motion',
                                 contents=['/world/**', f'- /{motion.robot.instance_path(motion.description.instance_id)}/collision/**'])
        tasks = rrb.Vertical(
            rrb.TextDocumentView(origin='live/task/current', name='Current Task'),
            rrb.Horizontal(rrb.TextDocumentView(origin='live/task/subtask', name='Sub Task'),
                           rrb.TextDocumentView(origin='live/task/action', name='Current Action')),
            row_shares=[2, 1])
        left = rrb.Vertical(world, tasks, row_shares=[3, 1])
        views = []
        for stream in streams:
            views.append(rrb.Vertical(
                rrb.Spatial2DView(origin=self.image_path(stream.key), name=stream.label),
                rrb.TextDocumentView(origin=self.status_path(stream.key), name=f'{stream.label} | Status'),
                row_shares=[3, 1]))
        content = rrb.Horizontal(left, rrb.Vertical(
            *(rrb.Horizontal(*views[i:i + 2]) for i in range(0, len(views), 2)))) if views else left
        recording.send_blueprint(rrb.Blueprint(
            rrb.Vertical(content, rrb.TextLogView(origin='live/task/events', name='Task Timeline | Runtime events'),
                         row_shares=[8, 1]), rrb.TimePanel(timeline=motion.timeline, state='collapsed'),
            rrb.BlueprintPanel(state='collapsed'), rrb.SelectionPanel(state='collapsed')))
        recording.set_time(motion.timeline, duration=0)
        recording.log('live/display', rr.AnyValues(process_id=os.getpid()))
        for key in self.streams:
            self._status(key, 'Disabled or no observation')
        recording.log('live/task/subtask', rr.TextDocument('Not connected'))
        recording.log('live/task/action', rr.TextDocument('Not connected'))
        self.task(TaskSnapshot(0))

    @staticmethod
    def image_path(key):
        return f'live/images/{segment(key)}'

    @staticmethod
    def status_path(key):
        return f'live/image_status/{segment(key)}'

    def _status(self, key, text):
        self.recording.log(self.status_path(key), self.rr.TextDocument(text))

    def advance(self, generation, at):
        if generation < self.generation:
            return False
        if generation > self.generation:
            self.generation = generation
            self.recording.set_time(self.motion.timeline, duration=at)
            self.recording.log('live/images', self.rr.Clear(recursive=True))
            self._image_times.clear()
            for key in self.streams:
                self._status(key, 'Waiting for this episode')
            # Other streams may arrive ahead of an earlier sampled SDK frame.
            # Reset on the first motion sample without changing its source clock.
            self._motion_reset_pending = True
        return True

    def motion_frames(self, frames):
        if self._motion_reset_pending:
            frames = [replace(frames[0], reset=True), *frames[1:]]
        self.motion.consume_batch(frames)
        self._motion_reset_pending = False

    def image(self, key, at, rgb):
        from PIL import Image

        if key not in self.streams:
            raise ValueError(f'Unknown image stream: {key}')
        if at <= self._image_times.get(key, -1):
            return
        elapsed(at)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Images must be nonempty HWC uint8 RGB arrays')
        picture = Image.fromarray(rgb)
        picture.thumbnail((self.max_edge, self.max_edge))
        self.recording.set_time(self.motion.timeline, duration=at)
        self.recording.log(self.image_path(key), self.rr.Image(np.asarray(picture)).compress(self.jpeg_quality))
        self._status(key, f'Observed at {at:.3f} s (not exposure time)')
        self._image_times[key] = at

    def expire(self, now):
        for key, at in tuple(self._image_times.items()):
            if now - at > self.stale_after:
                self.recording.set_time(self.motion.timeline, duration=now)
                self.recording.log(self.image_path(key), self.rr.Clear(recursive=True))
                self._status(key, 'Observation stream stale')
                del self._image_times[key]

    def task(self, snapshot):
        self.recording.set_time(self.motion.timeline, duration=snapshot.time)
        instruction = 'Task instruction unavailable' if snapshot.instruction is None else snapshot.instruction
        self.recording.log('live/task/current', self.rr.TextDocument(
            f'{instruction}\n\nSource: {snapshot.source or "Not provided"}\n'
            f'Runtime: {snapshot.status}\nEpisode: {snapshot.episode}'))
        for event in snapshot.events:
            if event.sequence > self._last_event:
                self.recording.set_time(self.motion.timeline, duration=event.time)
                expected = max(1, self._last_event + 1)
                if event.sequence > expected:
                    self.recording.log('live/task/events', self.rr.TextLog(
                        f'Runtime event gap: {event.sequence - expected} missing'))
                self.recording.log('live/task/events', self.rr.TextLog(event.text))
                self._last_event = event.sequence
