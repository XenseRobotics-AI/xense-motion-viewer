"""Rerun output adapter. SDK import and geometry loading occur only on use."""

import numpy as np

from .contracts import AttachedMesh, RobotDescription, RobotState
from .urdf import URDFModel


def segment(value):
    import rerun as rr
    return rr.escape_entity_path_part(value)


class RerunRenderer:
    """One renderer per RecordingStream, any number of robot instances.

    All instances share one elapsed-time clock, named by ``timeline``. Callers
    synchronize SDK clocks before logging. A new run/reset uses a new recording
    or an explicit monotonic elapsed clock; retrograde samples are rejected.
    Geometry and local offsets are static, link/TCP/target poses are temporal.
    """
    def __init__(self, recording, *, timeline='simulation', collision_visible=False, compact=False):
        import rerun as rr
        if not isinstance(timeline, str) or not timeline:
            raise ValueError("timeline must be a nonempty clock-domain name")
        self.rr = rr
        self.recording = recording
        self.timeline = timeline
        self.collision_visible = collision_visible
        self.compact = bool(compact)
        self._last_poses = {}
        self._spatial_view = None
        self.collision_batches = {}
        self._models = {}
        self._geometries = {}
        self._last_times = {}
        self._frames = {}
        self.geometry_log_count = 0
        self.update_count = 0
        recording.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP,
                      rr.CoordinateFrame(self.frame_id('world')), static=True)
        self.set_collision_visible(collision_visible)

    @staticmethod
    def instance_path(instance_id):
        return f'world/robots/{segment(instance_id)}'

    def link_path(self, instance_id, link):
        return f'{self.instance_path(instance_id)}/links/{segment(link)}'

    @staticmethod
    def frame_id(path):
        return f'tf#/{path}'

    def _transform(self, path, pose, parent='world', *, static=False):
        self.recording.log(path, self.rr.Transform3D(
            translation=pose[:3, 3], mat3x3=pose[:3, :3],
            parent_frame=self.frame_id(parent), child_frame=self.frame_id(path),
            relation=self.rr.TransformRelation.ParentFromChild,
        ), static=static)

    def add_robot(self, description: RobotDescription):
        instance = description.instance_id
        if instance in self._models:
            raise ValueError(f"Duplicate robot instance_id: {instance!r}")
        model = URDFModel(description)
        geometry = model.load_geometry()  # Validate/load everything before writing.
        if self.compact:
            geometry = self._batch_collisions(geometry, instance)
        self._models[instance] = model
        self._geometries[instance] = geometry
        self._frames[instance] = set()
        self.refresh_static(instance)
        self.set_collision_visible(self.collision_visible, force=True)
        return model

    def refresh_static(self, instance):
        """Refresh a late subscriber after bounded transport history eviction."""
        model = self._models[instance]
        geometry = self._geometries[instance]
        self.recording.log('world', self.rr.ViewCoordinates.RIGHT_HAND_Z_UP,
                           self.rr.CoordinateFrame(self.frame_id('world')), static=True)
        root = self.instance_path(instance)
        for layer, mesh in geometry:
            triangles = mesh.vertices[mesh.triangles]
            face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals = np.zeros_like(mesh.vertices)
            for corner in range(3):
                np.add.at(normals, mesh.triangles[:, corner], face_normals)
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            np.divide(normals, lengths, out=normals, where=lengths > 0)
            path = f'{root}/{layer}/{segment(mesh.name)}'
            self.recording.log(path, self.rr.Mesh3D(
                vertex_positions=mesh.vertices, triangle_indices=mesh.triangles, vertex_normals=normals,
                albedo_factor=mesh.rgba), self.rr.CoordinateFrame(self.frame_id(path)), static=True)
            self._transform(path, mesh.link_from_mesh, self.link_path(instance, mesh.parent_link), static=True)
            self.geometry_log_count += 1
        if self.compact:
            # Fixed link and declared tool placements are truly static relative
            # to their parent. Never place a static value on a dynamic entity.
            for joint in model.ordered_joints:
                if joint.kind == 'fixed':
                    self._transform(self.link_path(instance, joint.child), joint.origin,
                                    self.link_path(instance, joint.parent), static=True)
            for key, tool in model.description.tool_frames.items():
                path = f'{root}/tools/{segment(key)}'
                self._transform(path, tool.link_from_tool, self.link_path(instance, tool.parent_link), static=True)
                self.recording.log(path, self.rr.TransformAxes3D(0.025),
                                   self.rr.CoordinateFrame(self.frame_id(path)), static=True)

    def _batch_collisions(self, geometry, instance):
        """Exact triangle concatenation by rigid parent/color; no simplification."""
        groups = {}
        result = [(layer, mesh) for layer, mesh in geometry if layer != 'collision']
        for layer, mesh in geometry:
            if layer == 'collision':
                groups.setdefault((mesh.parent_link, mesh.rgba), []).append(mesh)
        for index, ((parent, rgba), meshes) in enumerate(groups.items()):
            vertices, faces, offset = [], [], 0
            for mesh in meshes:
                pose = mesh.link_from_mesh
                vertices.append(mesh.vertices @ pose[:3, :3].T + pose[:3, 3])
                faces.append(mesh.triangles + offset)
                offset += len(mesh.vertices)
            name = f'batch:{index}'
            result.append(('collision', AttachedMesh(
                name, parent, np.concatenate(vertices), np.concatenate(faces), rgba=rgba)))
            self.collision_batches[f'{instance}/{name}'] = tuple(mesh.name for mesh in meshes)
        return tuple(result)

    def _changed_transform(self, path, pose, parent='world'):
        previous = self._last_poses.get(path)
        if previous is None or not np.array_equal(previous, pose):
            self._transform(path, pose, parent)
            self._last_poses[path] = pose.copy()

    def update(self, instance_id: str, state: RobotState):
        model = self._models[instance_id]
        if state.time < self._last_times.get(instance_id, -1):
            raise ValueError("Time moved backwards; use a new recording for a new clock epoch")
        if self.compact:
            return self._update_compact(instance_id, state, model)
        transforms = model.link_transforms(state.joint_positions, state.base_transform)
        root = self.instance_path(instance_id)
        current_frames = {}
        for key, tool in model.description.tool_frames.items():
            current_frames[f'{root}/tools/{segment(key)}'] = (transforms[tool.parent_link] @ tool.link_from_tool, 0.06)
        for group, poses, size in [('tcp', state.tcp_poses, 0.08), ('targets', state.targets, 0.12)]:
            for key, pose in poses.items():
                current_frames[f'{root}/{group}/{segment(key)}'] = (pose, size)
        self.recording.set_time(self.timeline, duration=state.time)
        for link, pose in transforms.items():
            self._transform(self.link_path(instance_id, link), pose)
        for path in self._frames[instance_id] - current_frames.keys():
            self.recording.log(path, self.rr.Clear(recursive=True))
        for path, (pose, size) in current_frames.items():
            self._transform(path, pose)
            self.recording.log(path, self.rr.TransformAxes3D(size), self.rr.CoordinateFrame(self.frame_id(path)))
        self._frames[instance_id] = set(current_frames)
        self._last_times[instance_id] = state.time
        self.update_count += 1

    def _update_compact(self, instance, state, model):
        self.recording.set_time(self.timeline, duration=state.time)
        root = self.instance_path(instance)
        base = model.description.base_transform if state.base_transform is None else state.base_transform
        self._changed_transform(self.link_path(instance, model.root_link), base)
        values = model.resolve_positions(state.joint_positions)
        for joint in model.ordered_joints:
            if joint.kind != 'fixed':
                self._changed_transform(self.link_path(instance, joint.child), joint.placement(values[joint.name]),
                                        self.link_path(instance, joint.parent))
        current = {}
        for group, poses, size in [('tcp', state.tcp_poses, 0.035), ('targets', state.targets, 0.045)]:
            for key, pose in poses.items():
                current[f'{root}/{group}/{segment(key)}'] = (pose, size)
        for path in self._frames[instance] - current.keys():
            self.recording.log(path, self.rr.Clear(recursive=True))
            self._last_poses.pop(path, None)
        for path, (pose, size) in current.items():
            self._changed_transform(path, pose)
            if path not in self._frames[instance]:
                self.recording.log(path, self.rr.TransformAxes3D(size), self.rr.CoordinateFrame(self.frame_id(path)))
        self._frames[instance] = set(current)
        self.recording.log(f'{root}/sample', self.rr.AnyValues(elapsed=state.time))
        self._last_times[instance] = state.time
        self.update_count += 1

    def set_collision_visible(self, visible, *, force=False):
        """Toggle view filtering; static meshes are never resent."""
        import rerun.blueprint as rrb
        visible = bool(visible)
        if not force and self._spatial_view is not None and visible == self.collision_visible:
            return
        self.collision_visible = visible
        contents = ['/world/**']
        if not visible:
            contents.extend(f'- /{self.instance_path(instance)}/collision/**' for instance in self._models)
        if self._spatial_view is None:
            self._spatial_view = rrb.Spatial3DView(origin='world', contents=contents, name='Robot / TCP')
        else:
            # Keep the view ID and omit eye properties so a visibility toggle
            # does not replace the user's camera with a newly created view.
            self._spatial_view.contents = contents
        self.recording.send_blueprint(rrb.Blueprint(self._spatial_view))

    def flush(self):
        self.recording.flush()
