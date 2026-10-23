"""Rerun output adapter. SDK import and geometry loading occur only on use."""

import numpy as np

from .contracts import RobotDescription, RobotState
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
    def __init__(self, recording, *, timeline='simulation', collision_visible=False):
        import rerun as rr
        if not isinstance(timeline, str) or not timeline:
            raise ValueError("timeline must be a nonempty clock-domain name")
        self.rr = rr
        self.recording = recording
        self.timeline = timeline
        self.collision_visible = collision_visible
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
        self._models[instance] = model
        self._geometries[instance] = geometry
        self._frames[instance] = set()
        self.set_collision_visible(self.collision_visible)
        return model

    def update(self, instance_id: str, state: RobotState):
        model = self._models[instance_id]
        if state.time < self._last_times.get(instance_id, -1):
            raise ValueError("Time moved backwards; use a new recording for a new clock epoch")
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

    def set_collision_visible(self, visible):
        """Toggle view filtering; static meshes are never resent."""
        import rerun.blueprint as rrb
        self.collision_visible = bool(visible)
        contents = ['/world/**']
        if not visible:
            contents.extend(f'- /{self.instance_path(instance)}/collision/**' for instance in self._models)
        self.recording.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin='world', contents=contents)))

    def flush(self):
        self.recording.flush()
