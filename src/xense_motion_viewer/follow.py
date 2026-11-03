"""Offline hand input and deterministic command replay, independent of Qt/SDKs."""
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np

from .contracts import transform
from .control import InputOwnershipError, JointTarget, TcpTarget


def rotation_vector(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError('Rotation increment must be three finite radians')
    angle = np.linalg.norm(vector)
    if angle < 1e-14:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


class QtInputRoutes:
    """One Qt-owned lease, with mutually exclusive legacy and Follow producers.

    Switching pages may replace our own lease, never an external policy's lease.
    """
    def __init__(self, session):
        self.session = session
        self.mode = 'qt'
        self.lease = session.activate('qt')
        self.on_follow_submit = None
        self.legacy = _Route(self, 'qt')
        self.follow = _Route(self, 'follow')

    def select(self, mode):
        if mode not in ('qt', 'follow'):
            raise ValueError('Unknown Qt input mode')
        if mode != self.mode:
            try:
                self.lease = self.lease.transfer(mode)
            except InputOwnershipError:
                pass  # An external source keeps ownership even if it raced the page change.
        self.mode = mode

    def close(self):
        self.lease.close()


class _Route:
    def __init__(self, routes, mode):
        self.routes, self.mode = routes, mode

    @property
    def active(self):
        r = self.routes
        return r.mode == self.mode and r.lease.source == self.mode and r.lease.active

    def submit(self, command):
        if not self.active:
            raise InputOwnershipError(f'{self.mode} input is inactive')
        receipt = self.routes.lease.submit(command)
        if self.mode == 'follow' and self.routes.on_follow_submit:
            self.routes.on_follow_submit(receipt)
        return receipt

    @contextmanager
    def guard(self):
        lease = self.routes.lease
        with lease.guard():
            if self.routes.mode != self.mode or lease.source != self.mode:
                raise InputOwnershipError(f'{self.mode} input is inactive')
            yield


def command_data(command):
    if isinstance(command, TcpTarget):
        return dict(kind='tcp', tcp=command.tcp, pose=command.world_from_tcp.tolist())
    if isinstance(command, JointTarget):
        return dict(kind='joint', group=command.group, positions=dict(command.positions))
    raise TypeError('Unsupported Follow command')


def command_from_data(data):
    if data['kind'] == 'tcp':
        return TcpTarget(data['tcp'], data['pose'])
    if data['kind'] == 'joint':
        return JointTarget(data['group'], data['positions'])
    raise ValueError('Unknown Follow command kind')


@dataclass
class FollowTrace:
    metadata: dict
    initial_q: list
    initial_hands: dict
    period: float
    commands: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    duration_steps: int = 0
    end_reason: str = 'recording'
    failure: dict | None = None

    MAX_STEPS = 30_000
    MAX_COMMANDS = 30_000

    def save(self, path):
        self.validate()
        document = dict(schema='rmw-follow-v1', **vars(self))
        Path(path).write_text(json.dumps(document, ensure_ascii=False, allow_nan=False))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError('Follow recording exceeds 64 MiB')
        data = json.loads(path.read_text())
        if data.pop('schema', None) != 'rmw-follow-v1':
            raise ValueError('Unsupported Follow recording schema')
        trace = cls(**data)
        trace.validate()
        return trace

    def validate(self):
        if not isinstance(self.metadata, dict):
            raise ValueError('Missing recording metadata')
        if not np.isfinite(self.period) or not .004 <= self.period <= .01:
            raise ValueError('Invalid native period')
        if type(self.duration_steps) is not int or not 0 <= self.duration_steps <= self.MAX_STEPS:
            raise ValueError('Invalid recording duration')
        q = np.asarray(self.initial_q, dtype=float)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError('Invalid initial joints')
        if set(self.initial_hands) != {'left', 'right'}:
            raise ValueError('Recording needs both hand poses')
        for pose in self.initial_hands.values():
            transform(pose)
        if len(self.commands) > self.MAX_COMMANDS or len(self.samples) > self.MAX_STEPS:
            raise ValueError('Recording is too long')
        previous = -1
        for entry in self.commands:
            step = entry['step']
            if type(step) is not int or not max(0, previous) <= step < self.duration_steps:
                raise ValueError('Command steps must be ordered and inside the recording')
            previous = step
            command_from_data(entry['command'])
            if set(entry['hands']) != {'left', 'right'}:
                raise ValueError('Invalid hand snapshot')
            for pose in entry['hands'].values():
                transform(pose)
        if self.failure is not None:
            step = self.failure['step']
            if type(step) is not int or not 0 <= step < self.duration_steps:
                raise ValueError('Failed native attempt must be inside the recording')
            if self.failure['phase'] not in ('input', 'native', 'feedback'):
                raise ValueError('Invalid failure phase')
        elif self.end_reason.startswith('simulation fault:'):
            raise ValueError('Fault recording lacks the failed attempt index')
        # Feedback is diagnostic data, never a source of replay commands.
        json.dumps(vars(self), allow_nan=False)


class FollowInput:
    """Independent hand poses, grip anchors and 30 Hz absolute TCP targets.

    before_step/after_step run on the simulation thread, once per native tick.
    Mouse release holds the last target. Only an explicit new grip reanchors.
    """
    def __init__(self, inputs, read_tcp, *, period=.004, rate_hz=30.):
        if not np.isfinite(period) or not .004 <= period <= .01:
            raise ValueError('Invalid native period')
        if not np.isfinite(rate_hz) or not 0 < rate_hz <= 1 / period:
            raise ValueError('Invalid Follow sample rate')
        self.inputs, self.read_tcp = inputs, read_tcp
        self.period, self.rate_hz = period, rate_hz
        self.hands = {side: np.eye(4) for side in ('left', 'right')}
        self.anchors = {}
        self.targets = {}
        self.angular_rates = {side: np.zeros(3) for side in self.hands}
        self.step_index = 0
        self.next_sample = 0.
        self.enabled = False
        self.paused = ''
        self.message = '抓住小球建立基准；小球不会跟随末端反馈'
        self.recording = None
        self.trace = None
        self.playback = None
        self.play_capture = None
        self.play_index = 0

    @property
    def active(self):
        return self.enabled and self.inputs.active and not self.paused

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        self.clear_gestures()
        self.targets.clear()
        self.stop_playback('mode changed')
        self.stop_recording('mode changed')
        self.next_sample = self.step_index * self.period
        if enabled and not self.paused:
            self.message = '抓住小球建立基准；松手保留目标，机械臂反馈不会拉回小球'

    def clear_gestures(self):
        self.anchors.clear()
        for rate in self.angular_rates.values():
            rate.fill(0)

    def reset(self):
        self.stop_recording('simulation reset')
        self.clear_gestures()
        self.targets.clear()
        self.stop_playback('simulation reset')
        self.step_index, self.next_sample, self.paused = 0, 0., ''

    def begin_grip(self, side):
        if not self.active:
            raise InputOwnershipError('Follow input is inactive or simulation is paused')
        if self.playback:
            raise ValueError('先停止回放，再拖动小球')
        if side not in self.anchors:
            self.anchors[side] = (self.hands[side].copy(), transform(self.read_tcp(side)))
            self.targets[side] = self.anchors[side][1].copy()
            self.next_sample = min(self.next_sample, self.step_index * self.period)

    def end_grip(self, side):
        self.anchors.pop(side, None)
        self.angular_rates[side].fill(0)

    def move_hand(self, side, translation=(0., 0., 0.), rotation=(0., 0., 0.)):
        if not self.active or side not in self.anchors:
            raise InputOwnershipError('Grab the hand before moving it')
        delta = np.asarray(translation, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError('Hand translation must be three finite metres')
        pose = self.hands[side].copy()
        pose[:3, 3] += delta
        pose[:3, :3] = rotation_vector(rotation) @ pose[:3, :3]
        self.hands[side] = transform(pose).copy()
        self._update_target(side)

    def _update_target(self, side):
        hand, tcp = self.anchors[side]
        goal = tcp.copy()
        goal[:3, 3] += self.hands[side][:3, 3] - hand[:3, 3]
        goal[:3, :3] = self.hands[side][:3, :3] @ hand[:3, :3].T @ tcp[:3, :3]
        self.targets[side] = transform(goal).copy()

    def before_step(self):
        if not self.active:
            if not self.inputs.active:
                self.stop_recording('input ownership lost')
                self.stop_playback('input ownership lost')
                self.clear_gestures()
            return
        if self.playback:
            trace = self.playback
            while self.play_index < len(trace.commands) and trace.commands[self.play_index]['step'] == self.step_index:
                row = trace.commands[self.play_index]
                self.hands = {s: transform(p).copy() for s, p in row['hands'].items()}
                self.inputs.submit(command_from_data(row['command']))
                self.play_index += 1
            return
        for side in self.anchors:
            if np.any(self.angular_rates[side]):
                self.move_hand(side, rotation=self.angular_rates[side] * self.period)
        now = self.step_index * self.period
        if now + 1e-10 >= self.next_sample:
            for side, pose in self.targets.items():
                self.inputs.submit(TcpTarget(side, pose))
            self.next_sample += (int(max(0., now - self.next_sample) * self.rate_hz) + 1) / self.rate_hz

    def after_step(self, feedback=None):
        capture = self.recording or self.play_capture
        if capture and feedback is not None:
            capture.samples.append(dict(step=self.step_index, **feedback))
        self.step_index += 1
        if self.playback and self.step_index >= self.playback.duration_steps:
            self.stop_playback('completed')
            self.message = '回放结束；保留最后目标，仿真继续运行'
        if self.recording and self.step_index >= FollowTrace.MAX_STEPS:
            self.stop_recording('recording limit reached')
            self.message = '录制达到长度上限；跟随仍在运行'

    def record_accepted(self, receipt):
        capture = self.recording or self.play_capture
        if capture:
            capture.commands.append(dict(step=self.step_index, command=command_data(receipt.command),
                hands={s: p.tolist() for s, p in self.hands.items()}))
            if self.recording and len(capture.commands) >= FollowTrace.MAX_COMMANDS:
                self.stop_recording('command limit reached')

    def start_recording(self, metadata, initial_q):
        if not self.active:
            raise InputOwnershipError('Follow is not active')
        self.reset()
        self.trace = FollowTrace(metadata, list(initial_q), {s: p.tolist() for s, p in self.hands.items()}, self.period)
        self.recording = self.trace
        self.message = '正在记录输入与每个原生周期的反馈'

    def stop_recording(self, reason='stopped'):
        if self.recording:
            trace = self.recording
            self._finish_trace(trace, reason)
            self.recording = None

    def _finish_trace(self, trace, reason):
        trace.duration_steps = max(self.step_index, max((r['step'] + 1 for r in trace.commands), default=0),
                                   trace.failure['step'] + 1 if trace.failure else 0)
        trace.end_reason = reason

    def start_playback(self, trace, *, metadata=None):
        if not self.active:
            raise InputOwnershipError('Follow is not active')
        trace.validate()
        if trace.period != self.period:
            raise ValueError('Recording native period differs from this simulation')
        self.reset()
        self.hands = {s: transform(p).copy() for s, p in trace.initial_hands.items()}
        self.playback, self.play_index = trace, 0
        self.trace = FollowTrace(metadata or trace.metadata, list(trace.initial_q),
                                dict(trace.initial_hands), trace.period)
        self.play_capture = self.trace
        self.message = '按原始控制周期回放；结束后保留最后目标'
        if trace.duration_steps == 0:
            self.stop_playback('completed')

    def stop_playback(self, reason='stopped'):
        was_playing = self.playback is not None or self.play_capture is not None
        if self.play_capture:
            trace = self.play_capture
            self._finish_trace(trace, reason)
            self.play_capture = None
        self.playback = None
        self.targets.clear()
        if was_playing:
            self.message = '回放已停止提交；SDK 保留最后目标'

    def fault(self, error, *, phase='native'):
        self.paused = str(error)
        self.clear_gestures()
        capture = self.recording or self.play_capture
        if capture:
            capture.failure = dict(step=self.step_index, phase=phase, error=str(error))
        self.stop_playback('simulation fault: ' + str(error))
        self.stop_recording('simulation fault: ' + str(error))
        self.message = '仿真已暂停：' + str(error)
