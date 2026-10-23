"""Denormalized policy references and explicit one-step session submission.

No inference, clock synchronization, scheduler, IK, or hardware transport lives
here. All host times share the caller's elapsed clock. Offsets describe intent;
they never schedule work. Use this object on the robot SDK's owner thread.
"""
from dataclasses import dataclass
from threading import get_ident
from typing import Callable, Literal

import numpy as np

from .contracts import RobotState, frozen_array, name, transform
from .control import InputSession, JointTarget, SubmissionReceipt, TcpTarget

Origin = Literal['reference', 'simulated', 'measured']


def elapsed(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('Expected finite nonnegative elapsed time')
    return value


def origin(value):
    if value not in ('reference', 'simulated', 'measured'):
        raise ValueError('Origin must be reference, simulated, or measured')
    return value


@dataclass(frozen=True)
class SampleTime:
    """Host receipt is always known; source time needs an explicit clock name.

    Source and host clocks are not subtracted or silently synchronized.
    Freshness below means age since host receipt, not end-to-end latency.
    """
    received_at: float
    source_time: float | None = None
    source_clock: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'received_at', elapsed(self.received_at))
        if (self.source_time is None) != (self.source_clock is None):
            raise ValueError('Source time and clock must both be known or both absent')
        if self.source_time is not None:
            object.__setattr__(self, 'source_time', elapsed(self.source_time))
            name(self.source_clock)

    def freshness(self, now, max_age):
        age = elapsed(now) - self.received_at
        if age < 0:
            raise ValueError('Host now precedes receipt')
        return 'fresh' if age <= elapsed(max_age) else 'stale'


@dataclass(frozen=True, eq=False)
class PolicyChunk:
    """One complete absolute action chunk, already denormalized to radians/metres.

    ``values`` is [steps, named joints] or [steps, 4, 4]. Offsets are relative
    to host ``timing.received_at`` (the chunk availability epoch), not source
    time. Every step belongs to a single explicit SDK group. For TCP chunks,
    joint_names still declare that group's joints; frame must be world.
    Other action semantics require an external, explicit conversion.
    """
    chunk_id: str
    group: str
    joint_names: tuple[str, ...]
    time_offsets: np.ndarray
    values: np.ndarray
    timing: SampleTime
    expires_at: float
    kind: Literal['absolute_joint_positions', 'absolute_tcp_poses']
    frame: Literal['joint', 'world']
    tcp: str | None = None
    origin: Origin = 'reference'

    def __post_init__(self):
        name(self.chunk_id); name(self.group)
        names = tuple(name(n) for n in self.joint_names)
        if not names or len(names) != len(set(names)):
            raise ValueError('joint_names must be nonempty and unique')
        offsets, values = frozen_array(self.time_offsets), frozen_array(self.values)
        if (offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all()
                or offsets[0] < 0 or np.any(np.diff(offsets) <= 0)):
            raise ValueError('Offsets must be finite, nonnegative and strictly increasing')
        if not isinstance(self.timing, SampleTime):
            raise TypeError('timing must be SampleTime')
        expires = elapsed(self.expires_at)
        if expires <= self.timing.received_at:
            raise ValueError('Expiry must follow availability')
        if not np.isfinite(self.timing.received_at + offsets).all():
            raise ValueError('Intended host times must be finite')
        if self.kind == 'absolute_joint_positions':
            if self.frame != 'joint' or self.tcp is not None:
                raise ValueError('Joint positions require joint frame and no TCP')
            if values.shape != (len(offsets), len(names)) or not np.isfinite(values).all():
                raise ValueError('Joint values must be finite [steps, joint_names]')
        elif self.kind == 'absolute_tcp_poses':
            if self.frame != 'world':
                raise ValueError('TCP poses require explicit world frame')
            name(self.tcp)
            if values.shape != (len(offsets), 4, 4):
                raise ValueError('TCP values must be [steps, 4, 4]')
            for pose in values:
                transform(pose)
        else:
            raise ValueError('Only absolute joint positions and absolute TCP poses are supported')
        object.__setattr__(self, 'joint_names', names)
        object.__setattr__(self, 'time_offsets', offsets)
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'expires_at', expires)
        if self.origin != 'reference':
            raise ValueError('Policy predictions are references, never future measured samples')

    @classmethod
    def from_dict(cls, value):
        """Strict JSON-compatible constructor; unknown fields/semantics fail."""
        data = dict(value)
        data['timing'] = SampleTime(**data['timing'])
        return cls(**data)

    def target(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.values):
            raise ValueError('Selected step index is outside the chunk')
        if self.kind == 'absolute_joint_positions':
            return JointTarget(self.group, dict(zip(self.joint_names, self.values[index], strict=True)))
        return TcpTarget(self.tcp, self.values[index])

    def validate(self, capabilities):
        if self.group not in capabilities.groups or set(self.joint_names) != set(capabilities.groups[self.group]):
            raise ValueError('Chunk joint_names must exactly match the named SDK group')
        if self.tcp is not None and (self.tcp not in capabilities.tcp_targets or
                capabilities.tcp_targets[self.tcp].group != self.group):
            raise ValueError('TCP must belong to the declared SDK group')
        for index in range(len(self.values)):
            capabilities.validate(self.target(index))


@dataclass(frozen=True)
class StepRef:
    """Chunk association; None means no step was selected for a lifecycle event."""
    chunk_id: str
    step_index: int | None = None

    def __post_init__(self):
        name(self.chunk_id)
        if self.step_index is not None and (isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0):
            raise ValueError('step_index must be a nonnegative integer')


@dataclass(frozen=True, eq=False)
class JointPreview:
    """Optional SDK-provided full joint references, never measured future state.

    The provider owns Cartesian IK if supplied. Without this capability the
    renderer draws a TCP path only. No preview is sent to an actuator.
    """
    joint_names: tuple[str, ...]
    values: np.ndarray
    provider: str
    base_transform: np.ndarray | None = None

    def __post_init__(self):
        names = tuple(name(n) for n in self.joint_names)
        values = frozen_array(self.values)
        if not names or len(set(names)) != len(names) or values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError('Preview needs finite full named joint rows')
        name(self.provider)
        object.__setattr__(self, 'joint_names', names)
        object.__setattr__(self, 'values', values)
        if self.base_transform is not None:
            object.__setattr__(self, 'base_transform', transform(self.base_transform))


@dataclass(frozen=True)
class StreamEvent:
    """Immutable display data. An unavailable stream has no invented value.

    Telemetry association is supplied explicitly by the SDK/host, not inferred
    from timestamps. Feedback without an association is valid and stays so.
    """
    stage: str
    timing: SampleTime
    origin: Origin
    ref: StepRef | None = None
    chunk: PolicyChunk | None = None
    command: JointTarget | TcpTarget | None = None
    state: RobotState | None = None
    receipt: SubmissionReceipt | None = None
    preview: JointPreview | None = None
    reason: str = ''
    fallback: bool = False
    submission_sequence: int | None = None

    def __post_init__(self):
        if self.stage not in ('prediction', 'selection', 'submitted', 'superseded', 'expired', 'planned', 'sent', 'feedback'):
            raise ValueError('Unknown stream stage')
        origin(self.origin)
        if not isinstance(self.timing, SampleTime):
            raise TypeError('timing must be SampleTime')
        if self.ref is not None and not isinstance(self.ref, StepRef):
            raise TypeError('ref must be StepRef')
        if self.command is not None and not isinstance(self.command, (JointTarget, TcpTarget)):
            raise TypeError('command must be an immutable target')
        if self.stage == 'prediction':
            if not isinstance(self.chunk, PolicyChunk) or self.timing != self.chunk.timing or self.origin != self.chunk.origin:
                raise ValueError('Prediction needs its chunk availability time')
        elif self.chunk is not None or self.preview is not None:
            raise ValueError('Only predictions carry chunks/previews')
        if self.preview is not None and (not isinstance(self.preview, JointPreview) or len(self.preview.values) != len(self.chunk.values)):
            raise ValueError('Preview must contain every chunk step')
        if self.command is not None and self.stage not in ('planned', 'sent'):
            raise ValueError('Commands are reserved for planned/sent telemetry; submission uses its receipt')
        if self.stage in ('selection', 'submitted', 'superseded', 'expired') and self.ref is None:
            raise ValueError('Lifecycle/submission events require a chunk association')
        if self.stage in ('selection', 'submitted') and self.ref.step_index is None:
            raise ValueError('Selection/submission requires a selected step index')
        if self.stage == 'submitted':
            if not isinstance(self.receipt, SubmissionReceipt):
                raise ValueError('Submission needs a real session receipt')
        elif self.receipt is not None:
            raise ValueError('Only submitted events carry receipts')
        if self.state is not None and (self.stage != 'feedback' or not isinstance(self.state, RobotState)):
            raise ValueError('RobotState is reserved for feedback')
        if self.stage in ('planned', 'sent', 'feedback'):
            value = self.state if self.stage == 'feedback' else self.command
            if value is None and not self.reason:
                raise ValueError('Unavailable telemetry requires a reason')
            if self.stage in ('sent', 'feedback') and value is not None and self.origin == 'reference':
                raise ValueError('Actual samples require simulated or measured provenance')
        if not isinstance(self.fallback, bool):
            raise TypeError('fallback must be bool')
        if self.fallback and (self.stage != 'sent' or self.available or
                              self.origin != 'reference' or self.ref is None):
            raise ValueError('Fallback is only unavailable reference sent bookkeeping for a named step')
        if self.submission_sequence is not None and (isinstance(self.submission_sequence, bool) or
                not isinstance(self.submission_sequence, int) or self.submission_sequence < 1):
            raise ValueError('submission_sequence must be a positive session receipt sequence')

    @property
    def available(self):
        if self.stage == 'feedback':
            return self.state is not None
        if self.stage in ('planned', 'sent'):
            return self.command is not None
        return True


@dataclass(frozen=True)
class PolicySubmission:
    ref: StepRef
    receipt: SubmissionReceipt


class PolicyInput:
    """Explicit selection/execute over an existing Phase 2 producer lease.

    Accept/select/expire only publish data. Only execute_selected submits one
    target. Replacement/expiry invalidates selection, but never cancels a target
    already accepted by the SDK. Publisher must be a nonblocking data sink.
    No robot callback is ever handed to the visualization worker.
    """
    def __init__(self, inputs: InputSession, publish: Callable[[StreamEvent], object]):
        self.inputs, self.publish = inputs, publish
        self._owner = get_ident()
        self._active = None
        self._selected = None
        self._ids = set()
        self._time = 0.0
        self._closed = False
        self.display_drops = 0
        self.display_errors = 0
        self.last_display_error = None

    @property
    def active_chunk(self):
        return self._active

    @property
    def selected(self):
        return self._selected

    def _check(self, now, *, commit=True):
        if get_ident() != self._owner:
            raise RuntimeError('Policy input must be used on its SDK owner thread')
        if self._closed:
            raise RuntimeError('Policy input is closed')
        now = elapsed(now)
        if now < self._time:
            raise ValueError('Host time moved backwards')
        if commit:
            self._time = now
        return now

    def _emit(self, event):
        # Even a misbehaving display sink must not make an accepted target look
        # rejected. A custom sink must still honor the nonblocking contract.
        try:
            if self.publish(event) is False:
                self.display_drops += 1
        except Exception as exc:
            self.display_errors += 1
            self.last_display_error = f'{type(exc).__name__}: {exc}'

    def accept(self, chunk: PolicyChunk, *, preview: JointPreview | None = None):
        now = self._check(chunk.timing.received_at, commit=False)
        chunk.validate(self.inputs.controller.capabilities)
        if chunk.chunk_id in self._ids:
            raise ValueError('chunk_id cannot be reused in a policy input lifecycle')
        if preview is not None and (set(preview.joint_names) != set(self.inputs.controller.description.joint_names)
                or len(preview.values) != len(chunk.values)):
            raise ValueError('Preview must contain full robot joints for every step')
        self._time = now
        if self._active is not None:
            old = self._selected or StepRef(self._active.chunk_id)
            self._emit(StreamEvent('superseded', SampleTime(now), 'reference', old,
                                     reason=f'Replaced by {chunk.chunk_id}'))
        self._ids.add(chunk.chunk_id)
        self._active, self._selected = chunk, None
        self._emit(StreamEvent('prediction', chunk.timing, chunk.origin, chunk=chunk, preview=preview))

    def expire(self, now):
        now = self._check(now)
        if self._active is not None and now >= self._active.expires_at:
            ref = self._selected or StepRef(self._active.chunk_id)
            self._emit(StreamEvent('expired', SampleTime(now), 'reference', ref, reason='Host expiry reached'))
            self._active, self._selected = None, None

    def select(self, index, *, now):
        self.expire(now)
        if self._active is None:
            raise RuntimeError('No active unexpired chunk')
        self._active.target(index)
        self._selected = StepRef(self._active.chunk_id, int(index))
        self._emit(StreamEvent('selection', SampleTime(now), 'reference', self._selected))
        return self._selected

    def execute_selected(self, *, now):
        self.expire(now)
        if self._active is None or self._selected is None:
            raise RuntimeError('Select a step from an active unexpired chunk first')
        receipt = self.inputs.submit(self._active.target(self._selected.step_index))
        result = PolicySubmission(self._selected, receipt)
        self._emit(StreamEvent('submitted', SampleTime(now), 'reference', result.ref, receipt=receipt))
        # A setter return is not a send callback. A host may publish actual send
        # evidence separately, with the same ref and honest origin/source clock.
        self._emit(StreamEvent('sent', SampleTime(now), 'reference', result.ref,
                                 reason='Unavailable: target acceptance supplies no send telemetry', fallback=True,
                                 submission_sequence=receipt.sequence))
        return result

    def close(self):
        if get_ident() != self._owner:
            raise RuntimeError('Close on the SDK owner thread')
        if not self._closed:
            self.inputs.close()
            self._active, self._selected = None, None
            self._ids.clear()
            self._closed = True


def replay_events(events, consume):
    """Read-only replay: accepts immutable data and a display sink, no adapter."""
    for event in events:
        consume(event)
