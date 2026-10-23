"""Explicit offline reference movement for any scalar-joint URDF tree.

This example adapter is a rate-limited numerical follower, not a robot planner,
dynamics simulator, collision checker or hardware driver. advance() is called
by its owner; neither submit() nor snapshot() advances observations.
"""
from importlib.resources import files
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..contracts import RobotDescription, RobotState, ToolFrame
from ..control import ControlCapabilities, JointControl, JointTarget
from ..urdf import URDFModel, xyz_rpy


def demo_urdf():
    """Installed wheels contain this original primitive model; no vendor assets."""
    return Path(str(files('robot_motion_workbench').joinpath('assets/demo_seven.urdf')))


def description_from_urdf(urdf=None, *, instance_id='reference', tool_link=None,
                          package_roots=None, base_transform=None, link_from_tool=None):
    """Discover independent joints in URDF order, with an explicit tool parent.

    This convenience is for examples. Production adapters should supply their
    SDK's explicit joint mapping and calibrated transforms in RobotDescription.
    The example defaults deliberately use nonidentity world/base and tool poses.
    """
    path = demo_urdf() if urdf is None else Path(urdf)
    xml = ET.parse(path).getroot()
    names = tuple(e.attrib['name'] for e in xml.findall('joint')
                  if e.attrib['type'] != 'fixed' and e.find('mimic') is None)
    if not names:
        raise ValueError('The reference example needs at least one controllable joint')
    # Resolve/validate the graph before choosing its final descendant as a demo
    # tool parent. --tool-link overrides this convention for external models.
    base = xyz_rpy((0.3, -0.2, 0.12), (0.1, -0.15, 0.35)) if base_transform is None else base_transform
    desc = RobotDescription(instance_id, path, names, base, package_roots=package_roots or {})
    model = URDFModel(desc)
    parent = tool_link or model.ordered_joints[-1].child
    tool = xyz_rpy((0.04, -0.02, 0.12), (0.2, -0.1, 0.3)) if link_from_tool is None else link_from_tool
    return RobotDescription(instance_id, path, names, base,
                            tool_frames={'tool': ToolFrame(parent, tool)},
                            package_roots=package_roots or {})


def urdf_joint_controls(description):
    """URDF static limits for the offline example, never hardware validation."""
    xml = ET.parse(description.urdf_path).getroot()
    joints = {e.attrib['name']: e for e in xml.findall('joint')}
    controls = {}
    for n in description.joint_names:
        joint = joints[n]
        unit = 'm' if joint.attrib['type'] == 'prismatic' else 'rad'
        limit = joint.find('limit')
        bounded = limit is not None and joint.attrib['type'] != 'continuous'
        controls[n] = JointControl(unit,
            float(limit.attrib['lower']) if bounded and 'lower' in limit.attrib else None,
            float(limit.attrib['upper']) if bounded and 'upper' in limit.attrib else None,
            'mm' if unit == 'm' else 'deg', 1000 if unit == 'm' else 180 / np.pi)
    return controls


class ReferenceRobotAdapter:
    """Caller-stepped synthetic observations through the public robot interface."""
    def __init__(self, description, *, initial_positions=None, speed=0.5):
        self._description = description
        self.model = URDFModel(description)
        self._capabilities = ControlCapabilities(urdf_joint_controls(description),
                                                {'arm': description.joint_names})
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('speed must be finite and positive (SI units / second)')
        self.speed = float(speed)
        if initial_positions is None:
            initial_positions = {}
            for i, (n, spec) in enumerate(self._capabilities.joints.items()):
                value = (0.025 if spec.unit == 'm' else 0.65) * np.sin(i * 1.1 + 0.3)
                initial_positions[n] = np.clip(value,
                    spec.lower if spec.lower is not None else -np.inf,
                    spec.upper if spec.upper is not None else np.inf)
        command = self._capabilities.validate(JointTarget('arm', initial_positions))
        self._q = dict(command.positions)
        self._target = dict(self._q)
        self.time = 0.0

    def describe(self):
        return self._description

    def capabilities(self):
        # No Cartesian targets: this adapter has no IK.
        return self._capabilities

    def submit(self, command):
        command = self._capabilities.validate(command)
        self._target = dict(command.positions)

    def advance(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        for n in self._q:
            self._q[n] += float(np.clip(self._target[n] - self._q[n], -self.speed * dt, self.speed * dt))
        self.time += float(dt)

    def snapshot(self):
        placements = self.model.link_transforms(self._q)
        poses = {n: placements[t.parent_link] @ t.link_from_tool
                 for n, t in self._description.tool_frames.items()}
        return RobotState(self.time, self._q, tcp_poses=poses)
