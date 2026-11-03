"""Pose decoding shared by recorded and live action sources."""
import numpy as np


def poses_from_xyz_rotation6d(values):
    """Decode [..., XYZ, rotation column 1, column 2] in metres to SE(3).

    Gram-Schmidt orthogonalization matches the Elite policy convention.
    Degenerate predictions are rejected instead of drawing an invented pose.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError('Expected finite [steps, 9] XYZ / column rotation-6D')
    first = values[:, 3:6]
    norm = np.linalg.norm(first, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate first rotation-6D column')
    first = first / norm
    second = values[:, 6:9] - np.sum(first * values[:, 6:9], axis=1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Degenerate second rotation-6D column')
    second = second / norm
    poses = np.tile(np.eye(4), (len(values), 1, 1))
    poses[:, :3, :3] = np.stack((first, second, np.cross(first, second)), axis=-1)
    poses[:, :3, 3] = values[:, :3]
    return poses
