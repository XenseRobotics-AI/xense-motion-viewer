"""Read-only live display. This API has no controller, IK or SDK dependency."""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from .contracts import RobotDescription, RobotState, frozen_array, name, transform
from .policy import elapsed
from .policy_renderer import _FeedbackTrail
from .rerun_compat import time_panel
from .rerun_renderer import RerunRenderer, segment
from .streams import VisualizationWorker


@dataclass(frozen=True, eq=False)
class FutureTrajectory:
    """Remaining world-frame TCP references, identified by stable action IDs."""
    step_ids: tuple[str, ...]
    poses: np.ndarray

    def __post_init__(self):
        ids = tuple(name(value) for value in self.step_ids)
        poses = frozen_array(self.poses)
        if len(set(ids)) != len(ids) or poses.shape != (len(ids), 4, 4):
            raise ValueError('Expected unique step IDs and matching [steps, 4, 4] poses')
        for pose in poses:
            transform(pose)
        object.__setattr__(self, 'step_ids', ids)
        object.__setattr__(self, 'poses', poses)


@dataclass(frozen=True)
class LiveFrame:
    """A complete display snapshot; missing state is explicitly unavailable.

    time is host elapsed time. sample_id identifies a fresh SDK state sample;
    re-publishing the same sample never extends its measured TCP history.
    reset clears references and histories before applying this frame.
    """
    time: float
    futures: Mapping[str, FutureTrajectory] = field(default_factory=dict)
    state: RobotState | None = None
    sample_id: str | None = None
    reset: bool = False
    status: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'time', elapsed(self.time))
        futures = {name(key): value for key, value in self.futures.items()}
        if not all(isinstance(value, FutureTrajectory) for value in futures.values()):
            raise TypeError('futures must contain FutureTrajectory values')
        if self.state is not None and not isinstance(self.state, RobotState):
            raise TypeError('state must be RobotState or None')
        if self.sample_id is not None:
            name(self.sample_id)
        object.__setattr__(self, 'futures', MappingProxyType(futures))


class LiveRerunRenderer:
    """Single-thread consumer for live references and measured state only."""
    def __init__(self, recording, description: RobotDescription, *, timeline='host_elapsed'):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr, self.recording, self.description = rr, recording, description
        self.timeline = timeline
        self.robot = RerunRenderer(recording, timeline=timeline, compact=True)
        self.rr = self.robot.rr
        self.robot.add_robot(description)
        self.root = f'world/live/{segment(description.instance_id)}'
        self._futures = {}
        self._trails = {}
        self._sample_id = None
        self._last_time = -1.0
        self._missing = True
        recording.send_blueprint(rrb.Blueprint(
            rrb.Spatial3DView(origin='world', name='Future reference / measured TCP and history',
                              contents=['/world/**', f'- /{self.robot.instance_path(description.instance_id)}/collision/**']),
            time_panel(timeline, follow=True)))

    def _future(self, group, trajectory):
        path = f'{self.root}/future/{segment(group)}'
        previous = self._futures.get(group)
        ids = trajectory.step_ids
        if not ids:
            if previous is not None:
                self.recording.log(path, self.rr.Clear(recursive=True))
                self._futures.pop(group, None)
            return
        # Consumption removes a prefix. Keep immutable coordinates and only
        # change visibility; replacement/blending sends the new coordinates.
        if previous is not None:
            old_ids, old_poses = previous
            offset = len(old_ids) - len(ids)
            if (offset >= 0 and old_ids[offset:] == ids
                    and np.array_equal(old_poses[offset:], trajectory.poses)):
                mask = np.arange(len(old_ids)) >= offset
                self.recording.log(f'{path}/steps', self.rr.Points3D.from_fields(radii=np.where(mask, .006, 0.)))
                self.recording.log(f'{path}/path', self.rr.LineStrips3D.from_fields(
                    radii=np.where(mask[:-1] & mask[1:], .005, 0.)))
                return
        points = trajectory.poses[:, :3, 3]
        frame = self.rr.CoordinateFrame(self.robot.frame_id('world'))
        strips = np.stack((points[:-1], points[1:]), axis=1)
        self.recording.log(f'{path}/steps', self.rr.Points3D(points, colors=[100, 164, 255],
                                                          radii=.006, show_labels=False), frame)
        self.recording.log(f'{path}/path', self.rr.LineStrips3D(strips, colors=[100, 164, 255], radii=.005), frame)
        self._futures[group] = ids, trajectory.poses

    def _history(self, state):
        frame = self.rr.CoordinateFrame(self.robot.frame_id('world'))
        for key, pose in state.tcp_poses.items():
            trail = self._trails.setdefault(key, _FeedbackTrail())
            trail.append(pose[:3, 3])
            path = f'{self.root}/measured/{segment(key)}'
            for slot in sorted(trail.dirty):
                block_path = f'{path}/history/{slot}'
                if slot in trail.logged:
                    self.recording.log(block_path, self.rr.LineStrips3D.from_fields(strips=[trail.blocks[slot]]))
                else:
                    self.recording.log(block_path, self.rr.LineStrips3D([trail.blocks[slot]],
                        colors=[65, 240, 120], radii=.003), frame)
                    trail.logged.add(slot)
            trail.dirty.clear()
            self.recording.log(f'{path}/current', self.rr.Points3D([pose[:3, 3]],
                colors=[65, 240, 120], radii=.008, show_labels=False), frame)

    def consume_batch(self, frames):
        latest_state = None
        for frame in frames:
            if frame.time < self._last_time:
                raise ValueError('Live display host time moved backwards')
            self._last_time = frame.time
            self.recording.set_time(self.timeline, duration=frame.time)
            if frame.reset:
                self.recording.log(self.root, self.rr.Clear(recursive=True))
                self._futures.clear()
                self._trails.clear()
                self._sample_id = None
                self._missing = True
                latest_state = None
            if frame.state is None:
                if not self._missing:
                    # Preserve older segments, but do not connect across a
                    # missing/stale state interval or show a stale current dot.
                    for key, trail in self._trails.items():
                        trail.slot = (trail.slot + 1) % 32
                        trail.blocks[trail.slot] = []
                        self.recording.log(f'{self.root}/measured/{segment(key)}/current', self.rr.Clear(recursive=True))
                self._missing = True
                latest_state = None
            elif frame.sample_id is None or frame.sample_id != self._sample_id or self._missing:
                self._history(frame.state)
                self._sample_id, self._missing = frame.sample_id, False
                latest_state = replace(frame.state, time=frame.time)
        if not frames:
            return
        frame = frames[-1]
        if latest_state is not None:
            self.robot.update(self.description.instance_id, latest_state)
        self.recording.set_time(self.timeline, duration=frame.time)
        for group in self._futures.keys() - frame.futures.keys():
            self.recording.log(f'{self.root}/future/{segment(group)}', self.rr.Clear(recursive=True))
        self._futures = {key: value for key, value in self._futures.items() if key in frame.futures}
        for group, future in frame.futures.items():
            self._future(group, future)
        self.recording.log('live/status', self.rr.TextDocument(
            f'Blue: remaining policy reference. Green: measured TCP history / current point.\n'
            f'State: {"unavailable" if self._missing else "measured"}. {frame.status}'))

    def close(self):
        self.recording.flush()
        self.recording.disconnect()


class LiveMotionDisplay:
    """Bounded asynchronous display; publishing never waits for rendering."""
    def __init__(self, recording, description, *, capacity=8):
        self.renderer = LiveRerunRenderer(recording, description)
        self.worker = VisualizationWorker(self.renderer.consume_batch, capacity=capacity,
                                          close_callback=self.renderer.close, batch=True)

    def publish(self, frame: LiveFrame):
        return self.worker.publish(frame)

    @property
    def stats(self):
        return self.worker.stats

    def close(self, *, timeout=5.0):
        return self.worker.close(timeout=timeout)
