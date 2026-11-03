"""Read-only visualization boundary. Units: metres, radians, elapsed seconds.

Transforms are parent_from_child, acting on column vectors. Snapshots never
command a robot; the robot SDK owns DOF, IK, limits and the control lifecycle.
"""
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np


def name(value: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("Names must be nonempty strings without control characters")
    return value


def frozen_array(value, *, dtype=float):
    array = np.array(value, dtype=dtype, copy=True)
    # Back with immutable bytes: even setflags(write=True) cannot mutate a snapshot.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def transform(value):
    result = frozen_array(value)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("Expected finite 4x4 SE(3) transform")
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1, atol=1e-7, rtol=0)):
        raise ValueError("Transform must be rigid and right-handed")
    return result


def positions(values):
    result = {name(k): float(v) for k, v in values.items()}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("Joint values must be finite")
    return MappingProxyType(result)


def frames(values):
    return MappingProxyType({name(k): transform(v) for k, v in values.items()})


@dataclass(frozen=True, eq=False)
class ToolFrame:
    parent_link: str
    link_from_tool: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        name(self.parent_link)
        object.__setattr__(self, "link_from_tool", transform(self.link_from_tool))


@dataclass(frozen=True, eq=False)
class AttachedMesh:
    """Optional SDK collision envelope, in its own local coordinates."""
    name: str
    parent_link: str
    vertices: np.ndarray
    triangles: np.ndarray
    link_from_mesh: np.ndarray = field(default_factory=lambda: np.eye(4))
    rgba: tuple[float, float, float, float] = (0.15, 0.65, 0.6, 0.3)

    def __post_init__(self):
        name(self.name)
        name(self.parent_link)
        vertices = frozen_array(self.vertices)
        faces = np.asarray(self.triangles)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices must be finite Nx3")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in 'iu' or np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Mesh triangles must be valid integer vertex indices")
        rgba = tuple(float(v) for v in self.rgba)
        if len(rgba) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in rgba):
            raise ValueError("RGBA must contain four values in [0,1]")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", frozen_array(faces, dtype=np.uint32))
        object.__setattr__(self, "link_from_mesh", transform(self.link_from_mesh))
        object.__setattr__(self, "rgba", rgba)


@dataclass(frozen=True, eq=False)
class RobotDescription:
    instance_id: str
    urdf_path: Path
    joint_names: tuple[str, ...]
    base_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    joint_defaults: Mapping[str, float] = field(default_factory=dict)
    tool_frames: Mapping[str, ToolFrame] = field(default_factory=dict)
    collision_meshes: tuple[AttachedMesh, ...] = ()
    package_roots: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        name(self.instance_id)
        names = tuple(name(n) for n in self.joint_names)
        if len(set(names)) != len(names) or set(names) & set(self.joint_defaults):
            raise ValueError("Joint names must be unique and disjoint from defaults")
        tools = {name(k): v for k, v in self.tool_frames.items()}
        if not all(isinstance(v, ToolFrame) for v in tools.values()):
            raise TypeError("tool_frames values must be ToolFrame")
        meshes = tuple(self.collision_meshes)
        if not all(isinstance(m, AttachedMesh) for m in meshes) or len({m.name for m in meshes}) != len(meshes):
            raise ValueError("Collision meshes must have unique names")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "urdf_path", Path(self.urdf_path).expanduser().resolve(strict=True))
        object.__setattr__(self, "base_transform", transform(self.base_transform))
        object.__setattr__(self, "joint_defaults", positions(self.joint_defaults))
        object.__setattr__(self, "tool_frames", MappingProxyType(tools))
        object.__setattr__(self, "collision_meshes", meshes)
        object.__setattr__(self, "package_roots", MappingProxyType({name(k): Path(v).resolve() for k, v in self.package_roots.items()}))

    @property
    def dof(self):
        return len(self.joint_names)


@dataclass(frozen=True, eq=False)
class RobotState:
    """Complete snapshot; TCP and target transforms are world_from_frame.

    base_transform overrides the description's world_from_URDF_root. Omitted
    TCP/target frames disappear at this sample; they are not retained as stale.
    Joint keys must exactly match description.joint_names when rendered.
    """
    time: float
    joint_positions: Mapping[str, float]
    base_transform: np.ndarray | None = None
    tcp_poses: Mapping[str, np.ndarray] = field(default_factory=dict)
    targets: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("time must be finite nonnegative elapsed seconds")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "joint_positions", positions(self.joint_positions))
        if self.base_transform is not None:
            object.__setattr__(self, "base_transform", transform(self.base_transform))
        object.__setattr__(self, "tcp_poses", frames(self.tcp_poses))
        object.__setattr__(self, "targets", frames(self.targets))


class RobotAdapter(Protocol):
    """SDK-owned robot exposes descriptions and read-only snapshots."""
    def describe(self) -> RobotDescription: ...
    def snapshot(self) -> RobotState: ...
