"""Named, synchronous target submission; no IK, scheduling or driver lifecycle.

Commands use SI units. A successful submission only says the adapter accepted
the target; it says nothing about a hardware send, execution or achievement.
"""
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

import numpy as np

from .contracts import RobotAdapter, RobotDescription, RobotState, name, positions, transform


@dataclass(frozen=True)
class JointControl:
    """SDK limits in SI; display = SI * display_scale + display_offset.

    Bounds are optional for continuous joints. They validate input, and never
    replace the SDK's dynamic limits, jump checks, or collision checks.
    """
    unit: str = 'rad'
    lower: float | None = None
    upper: float | None = None
    display_unit: str = 'rad'
    display_scale: float = 1.0
    display_offset: float = 0.0

    def __post_init__(self):
        if self.unit not in ('rad', 'm'):
            raise ValueError('Joint SI unit must be rad or m')
        name(self.display_unit)
        for key in ('lower', 'upper', 'display_scale', 'display_offset'):
            value = getattr(self, key)
            if value is None and key in ('display_scale', 'display_offset'):
                raise ValueError(f'{key} must be finite')
            if value is not None:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f'{key} must be finite')
                object.__setattr__(self, key, value)
        if self.display_scale == 0:
            raise ValueError('display_scale cannot be zero')
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError('Joint lower limit must be below upper limit')

    def to_display(self, value):
        return float(value) * self.display_scale + self.display_offset

    def from_display(self, value):
        return (float(value) - self.display_offset) / self.display_scale


@dataclass(frozen=True)
class TcpControl:
    """An SDK-owned TCP controller for a named joint group, in world coordinates."""
    group: str

    def __post_init__(self):
        name(self.group)


@dataclass(frozen=True)
class ControlCapabilities:
    joints: Mapping[str, JointControl]
    groups: Mapping[str, tuple[str, ...]]
    tcp_targets: Mapping[str, TcpControl] = field(default_factory=dict)
    joint_targets: bool = True

    def __post_init__(self):
        joints = {name(k): v for k, v in self.joints.items()}
        if not all(isinstance(v, JointControl) for v in joints.values()):
            raise TypeError('joints values must be JointControl')
        groups = {name(k): tuple(name(n) for n in v) for k, v in self.groups.items()}
        for group, names in groups.items():
            if not names or len(set(names)) != len(names) or not set(names) <= joints.keys():
                raise ValueError(f'Group {group!r} needs unique known joint names')
        tcps = {name(k): v for k, v in self.tcp_targets.items()}
        if not all(isinstance(v, TcpControl) and v.group in groups for v in tcps.values()):
            raise ValueError('TCP controls must reference known groups')
        if not isinstance(self.joint_targets, bool):
            raise TypeError('joint_targets must be bool')
        object.__setattr__(self, 'joints', MappingProxyType(joints))
        object.__setattr__(self, 'groups', MappingProxyType(groups))
        object.__setattr__(self, 'tcp_targets', MappingProxyType(tcps))

    def validate(self, command):
        """Validate before adapter invocation and return canonical named order."""
        if isinstance(command, JointTarget):
            if not self.joint_targets:
                raise ValueError('Joint targets are not supported')
            if command.group not in self.groups:
                raise ValueError(f'Unknown control group: {command.group!r}')
            names = self.groups[command.group]
            missing = set(names) - command.positions.keys()
            unknown = command.positions.keys() - set(names)
            if missing or unknown:
                raise ValueError(f'Joint names must exactly match group; missing={sorted(missing)}, unknown={sorted(unknown)}')
            ordered = {}
            for joint in names:
                value, spec = command.positions[joint], self.joints[joint]
                if ((spec.lower is not None and value < spec.lower) or
                        (spec.upper is not None and value > spec.upper)):
                    raise ValueError(f'Joint {joint!r} target is outside SDK limits')
                ordered[joint] = value
            return JointTarget(command.group, ordered)
        if isinstance(command, TcpTarget):
            if command.tcp not in self.tcp_targets:
                raise ValueError(f'Unsupported TCP target: {command.tcp!r}')
            return TcpTarget(command.tcp, command.world_from_tcp)
        raise TypeError('Expected JointTarget or TcpTarget')


@dataclass(frozen=True)
class JointTarget:
    """Complete named positions for one control group; never an array by index."""
    group: str
    positions: Mapping[str, float]

    def __post_init__(self):
        name(self.group)
        object.__setattr__(self, 'positions', positions(self.positions))
        if not self.positions:
            raise ValueError('A joint target cannot be empty')


@dataclass(frozen=True, eq=False)
class TcpTarget:
    """Rigid world_from_tcp pose; reachability and IK remain with the SDK."""
    tcp: str
    world_from_tcp: np.ndarray

    def __post_init__(self):
        name(self.tcp)
        object.__setattr__(self, 'world_from_tcp', transform(self.world_from_tcp))


Command = JointTarget | TcpTarget


@dataclass(frozen=True)
class SubmissionReceipt:
    sequence: int
    source: str
    command: Command
    status: str = field(default='submitted', init=False)


class ControllableRobotAdapter(RobotAdapter, Protocol):
    def capabilities(self) -> ControlCapabilities: ...
    def submit(self, command: Command) -> None:
        """Accept a validated target or raise; never imply sent or achieved."""
        ...


class InputOwnershipError(RuntimeError):
    pass


class RobotSession:
    """One adapter, one active producer, synchronous serialized callbacks.

    activate(..., replace=True) is an explicit takeover, invalidating the old
    InputSession even if the source name is reused. No queue, servo loop, TTL,
    retries, implicit stepping, stop or hardware connection is provided.
    """
    def __init__(self, adapter: ControllableRobotAdapter):
        self._adapter = adapter
        self.description = adapter.describe()
        self.capabilities = adapter.capabilities()
        if set(self.capabilities.joints) != set(self.description.joint_names):
            raise ValueError('Capabilities must describe exactly the robot joint names')
        self._lock = RLock()
        self._generation = 0
        self._source = None
        self._sequence = 0
        self._last_receipt = None

    @property
    def active_source(self):
        with self._lock:
            return self._source

    @property
    def last_receipt(self):
        with self._lock:
            return self._last_receipt

    def activate(self, source: str, *, replace=False):
        name(source)
        with self._lock:
            if self._source is not None and not replace:
                raise InputOwnershipError(f'Input is owned by {self._source!r}; explicit takeover required')
            self._generation += 1
            self._source = source
            return InputSession(self, source, self._generation)

    def snapshot(self):
        with self._lock:
            state = self._adapter.snapshot()
            if not isinstance(state, RobotState) or set(state.joint_positions) != set(self.description.joint_names):
                raise ValueError('Adapter snapshot must contain exactly the described joints')
            return state

    def _is_active(self, producer):
        return self._source == producer.source and self._generation == producer._generation

    def _submit(self, producer, command):
        with self._lock:
            if not self._is_active(producer):
                raise InputOwnershipError('Input session is inactive; acquire ownership explicitly')
            canonical = self.capabilities.validate(command)
            self._adapter.submit(canonical)
            self._sequence += 1
            self._last_receipt = SubmissionReceipt(self._sequence, producer.source, canonical)
            return self._last_receipt


@dataclass(frozen=True)
class InputSession:
    _controller: RobotSession = field(repr=False)
    source: str
    _generation: int = field(repr=False)

    @property
    def controller(self):
        return self._controller

    @property
    def active(self):
        with self._controller._lock:
            return self._controller._is_active(self)

    def submit(self, command: Command) -> SubmissionReceipt:
        return self._controller._submit(self, command)

    def close(self):
        with self._controller._lock:
            if self._controller._is_active(self):
                self._controller._source = None
                self._controller._generation += 1


class CallbackRobotAdapter:
    """Minimal external SDK bridge. The caller owns SDK construction/lifecycle.

    read_state returns RobotState; submit_target accepts a named Command and
    may raise. The callback must not interpret acceptance as hardware feedback.
    """
    def __init__(self, description: RobotDescription, capabilities: ControlCapabilities,
                 *, read_state: Callable[[], RobotState], submit_target: Callable[[Command], None]):
        self._description, self._capabilities = description, capabilities
        self._read_state, self._submit_target = read_state, submit_target

    def describe(self):
        return self._description

    def capabilities(self):
        return self._capabilities

    def snapshot(self):
        return self._read_state()

    def submit(self, command):
        self._submit_target(self._capabilities.validate(command))
