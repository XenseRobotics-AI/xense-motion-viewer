"""Display kinematics only: no IK, trajectory generation, limits or actuation."""
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import AttachedMesh, RobotDescription, name, transform


def vector(text, size, default):
    result = np.asarray(default if text is None else [float(v) for v in text.split()], dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite values")
    return result


def xyz_rpy(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    x, y, z = vector(None, 3, rpy)
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    result = np.eye(4)
    result[:3, :3] = [[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                      [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                      [-sy, cy*sx, cy*cx]]
    result[:3, 3] = vector(None, 3, xyz)
    return result


def origin(element):
    if element is None:
        return np.eye(4)
    return xyz_rpy(vector(element.get('xyz'), 3, [0, 0, 0]), vector(element.get('rpy'), 3, [0, 0, 0]))


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None

    def placement(self, value):
        motion = np.eye(4)
        if self.kind in ('revolute', 'continuous'):
            x, y, z = self.axis
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            motion[:3, :3] += np.sin(value) * skew + (1 - np.cos(value)) * (skew @ skew)
        elif self.kind == 'prismatic':
            motion[:3, 3] = self.axis * value
        return self.origin @ motion


class URDFModel:
    def __init__(self, description: RobotDescription):
        self.description = description
        root = ET.parse(description.urdf_path).getroot()
        if root.tag != 'robot':
            raise ValueError("Expected a URDF robot root")
        self.links = tuple(name(e.attrib['name']) for e in root.findall('link'))
        if not self.links or len(set(self.links)) != len(self.links):
            raise ValueError("URDF must contain unique links")
        self.joints = {}
        child_links = set()
        for e in root.findall('joint'):
            kind = e.attrib['type']
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError(f"Unsupported URDF joint type {kind!r}; provide scalar joints")
            n = name(e.attrib['name'])
            parent, child = e.find('parent').attrib['link'], e.find('child').attrib['link']
            if n in self.joints or parent not in self.links or child not in self.links or child in child_links:
                raise ValueError(f"Invalid URDF joint graph at {n!r}")
            child_links.add(child)
            axis_element = e.find('axis')
            axis = vector(None if axis_element is None else axis_element.get('xyz'), 3, [1, 0, 0])
            norm = np.linalg.norm(axis)
            if kind != 'fixed' and norm < 1e-12:
                raise ValueError(f"Zero joint axis: {n}")
            axis = axis / norm if norm else axis
            m = e.find('mimic')
            mimic = None if m is None else (m.attrib['joint'], float(m.get('multiplier', '1')), float(m.get('offset', '0')))
            if mimic is not None and (kind == 'fixed' or not np.isfinite(mimic[1:]).all()):
                raise ValueError(f"Invalid mimic joint {n}")
            self.joints[n] = Joint(n, kind, parent, child, origin(e.find('origin')), axis, mimic)
        roots = set(self.links) - child_links
        if len(roots) != 1:
            raise ValueError("URDF must be one connected tree")
        self.root_link = roots.pop()
        self.ordered_joints = []
        reached = {self.root_link}
        pending = list(self.joints.values())
        while pending:
            ready = [j for j in pending if j.parent in reached]
            if not ready:
                raise ValueError("Disconnected or cyclic URDF graph")
            for j in ready:
                if j.child in reached:
                    raise ValueError("Cyclic URDF graph")
                self.ordered_joints.append(j)
                reached.add(j.child)
                pending.remove(j)
        independent = {j.name for j in self.joints.values() if j.kind != 'fixed' and j.mimic is None}
        supplied = set(description.joint_names) | set(description.joint_defaults)
        if supplied != independent:
            raise ValueError(f"Independent URDF joints mismatch: missing={independent-supplied}, unknown={supplied-independent}")
        self.resolve_positions({n: 0 for n in description.joint_names})  # validate mimic references/cycles
        for frame in description.tool_frames.values():
            if frame.parent_link not in self.links:
                raise ValueError(f"Unknown tool parent link: {frame.parent_link}")
        for mesh in description.collision_meshes:
            if mesh.parent_link not in self.links:
                raise ValueError(f"Unknown collision parent link: {mesh.parent_link}")
        self._xml = root

    def resolve_positions(self, positions):
        if set(positions) != set(self.description.joint_names):
            raise ValueError("Snapshot joint keys must exactly match description.joint_names")
        values = dict(self.description.joint_defaults) | dict(positions)
        active = set()
        def resolve(n):
            if n in values:
                return values[n]
            if n in active or n not in self.joints:
                raise ValueError(f"Cyclic or missing mimic source: {n}")
            active.add(n)
            joint = self.joints[n]
            if joint.mimic is not None:
                source, scale, offset = joint.mimic
                values[n] = scale * resolve(source) + offset
            elif joint.kind == 'fixed':
                values[n] = 0.0
            else:
                raise ValueError(f"Missing joint value: {n}")
            active.remove(n)
            return values[n]
        for n in self.joints:
            resolve(n)
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Joint values must be finite")
        return values

    def link_transforms(self, positions, base_transform=None):
        values = self.resolve_positions(positions)
        base = self.description.base_transform if base_transform is None else transform(base_transform)
        result = {self.root_link: base}
        for j in self.ordered_joints:
            result[j.child] = result[j.parent] @ j.placement(values[j.name])
        return result

    def mesh_path(self, filename):
        uri = urlparse(filename)
        if uri.scheme == 'package':
            if uri.netloc not in self.description.package_roots:
                raise ValueError(f"Set package_roots for package://{uri.netloc}")
            path = self.description.package_roots[uri.netloc] / unquote(uri.path.lstrip('/'))
        elif uri.scheme == 'file':
            if uri.netloc not in ('', 'localhost'):
                raise ValueError("Remote file URIs are unsupported")
            path = Path(unquote(uri.path))
        elif uri.scheme:
            raise ValueError(f"Unsupported mesh URI scheme: {uri.scheme}")
        else:
            path = self.description.urdf_path.parent / filename
        return path.resolve(strict=True)

    def load_geometry(self):
        """Load each source mesh once, preserving scene-node transforms and units."""
        import trimesh
        source_cache = {}
        result = []
        materials = {}
        for m in self._xml.findall('material'):
            color = m.find('color')
            if color is not None:
                materials[m.get('name')] = vector(color.get('rgba'), 4, [0.7, 0.7, 0.7, 1])
        for link in self._xml.findall('link'):
            for layer in ('visual', 'collision'):
                if layer == 'collision' and self.description.collision_meshes:
                    continue  # SDK envelopes replace URDF collision shapes.
                for index, element in enumerate(link.findall(layer)):
                    geometry = element.find('geometry')
                    if geometry is None or len(geometry) != 1:
                        raise ValueError("Expected exactly one URDF geometry per visual/collision")
                    shape = geometry[0]
                    if shape.tag == 'mesh':
                        path = self.mesh_path(shape.attrib['filename'])
                        if path not in source_cache:
                            # Scene.to_mesh bakes each internal node transform, unlike concatenating raw geometry.
                            source_cache[path] = trimesh.load_scene(path, process=False).to_mesh()
                        mesh = source_cache[path].copy()
                        scale = vector(shape.get('scale'), 3, [1, 1, 1])
                        if np.any(scale == 0):
                            raise ValueError("Mesh scale must be nonzero")
                        mesh.apply_scale(scale)
                    elif shape.tag == 'box':
                        size = vector(shape.get('size'), 3, [0, 0, 0])
                        if np.any(size <= 0):
                            raise ValueError("Box sizes must be positive")
                        mesh = trimesh.creation.box(extents=size)
                    elif shape.tag in ('cylinder', 'sphere'):
                        radius = float(shape.attrib['radius'])
                        length = float(shape.get('length', '1'))
                        if not np.isfinite([radius, length]).all() or min(radius, length) <= 0:
                            raise ValueError("Primitive dimensions must be positive and finite")
                        mesh = (trimesh.creation.cylinder(radius=radius, height=length, sections=32)
                                if shape.tag == 'cylinder' else trimesh.creation.icosphere(subdivisions=2, radius=radius))
                    else:
                        raise ValueError(f"Unsupported URDF geometry: {shape.tag}")
                    rgba = np.array([0.7, 0.7, 0.7, 1.0])
                    material = element.find('material')
                    if material is not None:
                        color = material.find('color')
                        rgba = (vector(color.get('rgba'), 4, rgba) if color is not None
                                else materials.get(material.get('name'), rgba))
                    if layer == 'collision':
                        rgba = [0.15, 0.65, 0.6, 0.3]
                    result.append((layer, AttachedMesh(f"{link.attrib['name']}:{index}", link.attrib['name'],
                        mesh.vertices, mesh.faces, origin(element.find('origin')), tuple(rgba))))
        result.extend(('collision', mesh) for mesh in self.description.collision_meshes)
        self.source_mesh_count = len(source_cache)
        return tuple(result)
