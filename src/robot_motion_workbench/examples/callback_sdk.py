"""Executable minimal integration: replace two functions with SDK calls.

The caller supplies an existing SDK, URDF, limits and explicit SDK vector order.
Construction/start/stop, clocks, thread ownership and IK stay with that caller.
"""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np

from ..adapters.reference import description_from_urdf, urdf_joint_controls
from ..contracts import RobotState
from ..control import CallbackRobotAdapter, ControlCapabilities, JointTarget, RobotSession


def make_callback_session(description, joint_controls, *, sdk_joint_names,
                          read_positions, write_positions, clock=monotonic):
    """read_positions() -> SI vector; write_positions(list) accepts a target.

    A read is timed at host poll completion. No source timestamp, actual-send
    proof or calibrated TCP is invented. Calls run on the calling thread.
    """
    order = tuple(sdk_joint_names)
    if len(set(order)) != len(order) or set(order) != set(description.joint_names):
        raise ValueError('Explicit SDK vector order must map every described joint exactly once')
    capabilities = ControlCapabilities(joint_controls, {'arm': description.joint_names})
    epoch = clock()
    def read_state():
        q = np.array(read_positions(), dtype=float, copy=True)
        if q.shape != (len(order),) or not np.isfinite(q).all():
            raise ValueError('SDK feedback must contain one finite SI position per mapped joint')
        return RobotState(clock() - epoch, dict(zip(order, q, strict=True)))
    def submit_target(command):
        # This example advertises joint control only, so the session validates
        # and rejects TCP commands before this callback is reached.
        write_positions([command.positions[n] for n in order])
    return RobotSession(CallbackRobotAdapter(description, capabilities,
        read_state=read_state, submit_target=submit_target))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Executable callback mapping example with a fake external SDK; no hardware')
    parser.add_argument('--urdf', type=Path)
    args = parser.parse_args(argv)
    description = description_from_urdf(args.urdf)
    controls = urdf_joint_controls(description)  # Example only: verify SDK limits in a real integration.
    order = tuple(reversed(description.joint_names))  # Demonstrate nontrivial vector mapping.
    class ExternalSdkExample:
        def __init__(self):
            self.q = [float(np.clip(0., controls[n].lower if controls[n].lower is not None else -np.inf,
                controls[n].upper if controls[n].upper is not None else np.inf)) for n in order]
            self.accepted = []
        def read_positions(self):
            return self.q.copy()
        def accept_positions(self, target):
            self.accepted.append(target.copy())  # No motion or feedback is fabricated.
    sdk = ExternalSdkExample()
    session = make_callback_session(description, controls, sdk_joint_names=order,
        read_positions=sdk.read_positions, write_positions=sdk.accept_positions)
    initial = session.snapshot()
    inputs = session.activate('example')
    receipt = inputs.submit(JointTarget('arm', initial.joint_positions))
    assert session.snapshot().joint_positions == initial.joint_positions
    inputs.close()
    print(json.dumps(dict(status=receipt.status, sdk_order=order, accepted_vectors=sdk.accepted,
        observed_unchanged=True, source_sample_time=None, sent='unavailable', hardware_connections=0), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
