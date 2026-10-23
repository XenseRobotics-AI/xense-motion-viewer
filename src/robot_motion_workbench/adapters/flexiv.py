"""Optional mapping to an already configured libpyflexiv control object.

No vendor imports or lifecycle calls. The caller owns Robot, Enable, mode,
control start/stop, RT scheduling, calibration and safety. Inspected API:
libpyflexiv src/flexiv_rt.cpp PyJointImpedanceControl / JointState bindings.
"""
from time import monotonic

import numpy as np

from ..contracts import RobotState
from ..control import ControlCapabilities


class FlexivJointAdapter:
    """Map URDF names to SDK J1..J7, q [rad], and shared-memory target writes.

    sdk_joint_names is the caller's explicit seven-name SDK vector order; it
    need not equal description order. joint_controls must be caller-verified
    radian limits. No TCP controls/poses are inferred from uncalibrated SDK data.
    Snapshot time is elapsed host poll completion, NOT a source sample time;
    neither source timestamps nor actual-send telemetry exist in this API.
    """
    def __init__(self, control, description, *, sdk_joint_names, joint_controls,
                 group='arm', clock=monotonic):
        self._control, self._description = control, description
        self.sdk_joint_names = tuple(sdk_joint_names)
        if (len(self.sdk_joint_names) != 7 or len(set(self.sdk_joint_names)) != 7
                or set(self.sdk_joint_names) != set(description.joint_names)):
            raise ValueError('SDK order must contain exactly seven unique described joint names')
        if set(joint_controls) != set(description.joint_names):
            raise ValueError('Supply caller-verified controls for every described joint')
        if any(spec.unit != 'rad' for spec in joint_controls.values()):
            raise ValueError('Flexiv joint positions use radians')
        self._capabilities = ControlCapabilities(joint_controls, {group: description.joint_names})
        self._clock, self._epoch = clock, clock()

    def describe(self):
        return self._description

    def capabilities(self):
        return self._capabilities

    def is_running(self):
        return bool(self._control.is_running())

    def snapshot(self):
        if not self.is_running():
            raise RuntimeError('Caller-owned Flexiv control is not running')
        state = self._control.get_state()
        q = np.array(state.q, dtype=float, copy=True)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Flexiv JointState.q must contain seven finite radians')
        received_at = self._clock() - self._epoch
        return RobotState(received_at, dict(zip(self.sdk_joint_names, q, strict=True)))

    def submit(self, command):
        command = self._capabilities.validate(command)
        if not self.is_running():
            raise RuntimeError('Caller-owned Flexiv control is not running')
        # This SDK call writes shared memory. Return means submitted only; no
        # inference about transport send, RT consumption, or achieved state.
        self._control.set_target_joints([command.positions[n] for n in self.sdk_joint_names])
