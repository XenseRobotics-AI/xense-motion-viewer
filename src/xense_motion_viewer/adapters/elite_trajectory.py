"""Recorded Elite actions used as policy references, never as live feedback."""
import json
from pathlib import Path

import numpy as np

from ..contracts import frozen_array
from ..policy import PolicyChunk, SampleTime
from ..poses import poses_from_xyz_rotation6d


DEFAULT_TRAJECTORY = Path(__file__).resolve().parents[1] / 'assets' / 'elite_episode_000.npz'


def tcp_poses_from_actions(actions, feature_names, arm):
    """Decode world XYZ (metres) + the first two rotation-matrix columns.

    This is the acquisition driver's column convention, not Euler angles or
    the row convention used by some other rotation-6D implementations.
    """
    if arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    names = list(feature_names)
    if len(set(names)) != len(names):
        raise ValueError('Action feature names must be unique')
    values = np.asarray(actions, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
        raise ValueError('Actions must be finite rows matching feature_names')
    keys = [f'{arm}_tcp.{key}' for key in ('x', 'y', 'z', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6')]
    try:
        tcp = values[:, [names.index(key) for key in keys]]
    except ValueError as exc:
        raise ValueError(f'Missing {arm} TCP action features') from exc
    return poses_from_xyz_rotation6d(tcp)


class RecordedTrajectory:
    """A small, self-contained export of one LeRobot episode's action column."""
    def __init__(self, path=DEFAULT_TRAJECTORY, *, arm='left'):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.metadata = json.loads(str(archive['metadata'].item()))
            self.source_times = frozen_array(archive['timestamps'])
            frames = np.asarray(archive['frame_indices'])
            actions = np.asarray(archive['actions'])
        meta = self.metadata
        if (meta.get('format_version') != 1 or meta.get('robot_type') != 'bi_elite_cs66_rt'
                or meta.get('frame') != 'world' or meta.get('position_unit') != 'm'
                or meta.get('rotation_encoding') != 'first_two_columns'
                or meta.get('source_column') != 'action'):
            raise ValueError('Expected an Elite world-frame action export in metres / column rotation-6D')
        self.fps = float(meta['fps'])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('Recorded fps must be finite and positive')
        t = self.source_times
        if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or t[0] < 0
                or np.any(np.diff(t) <= 0)):
            raise ValueError('Recorded timestamps must be finite, nonnegative and strictly increasing')
        if (frames.shape != t.shape or frames.dtype.kind not in 'iu' or frames[0] < 0
                or np.any(np.diff(frames.astype(np.int64)) != 1)):
            raise ValueError('Recorded frame indices must be consecutive nonnegative integers')
        if actions.ndim != 2 or len(actions) != len(t) or meta.get('frame_count') != len(t):
            raise ValueError('Action, timestamp, frame and metadata lengths must agree')
        self.frame_indices = frozen_array(frames, dtype=np.int64)
        self.poses = frozen_array(tcp_poses_from_actions(actions, meta['feature_names'], arm))
        gripper_key = f'{arm}_gripper.pos'
        self.gripper_openings = None
        if gripper_key in meta['feature_names']:
            openings = actions[:, meta['feature_names'].index(gripper_key)]
            if not np.isfinite(openings).all() or np.any((openings < 0) | (openings > 1)):
                raise ValueError('Recorded gripper opening must be in [0, 1] (closed to open)')
            self.gripper_openings = frozen_array(openings)
        self.arm = arm
        self.times = frozen_array(t - t[0])
        # Hold the final sample for one acquisition frame, then stop (no loop).
        self.duration = float(self.times[-1] + 1 / self.fps)

    def index_at(self, elapsed):
        """Select the most recent recorded action; never select a future row."""
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Playback elapsed time must be finite and nonnegative')
        return max(0, min(int(np.searchsorted(self.times, elapsed, side='right')) - 1, len(self.times) - 1))

    def chunk(self, capabilities, start, *, now, size=60):
        """Publish the next complete reference chunk with original source time."""
        if not 0 <= start < len(self.times) or size < 1:
            raise ValueError('Invalid recorded chunk range')
        end = min(start + size, len(self.times))
        offsets = self.times[start:end] - self.times[start]
        episode = self.metadata['episode_index']
        chunk = PolicyChunk(
            f'recorded-episode-{episode}-{self.arm}-frame-{self.frame_indices[start]}',
            self.arm, capabilities.groups[self.arm], offsets, self.poses[start:end],
            SampleTime(now, float(self.source_times[start]), f'dataset_episode_{episode}_seconds'),
            now + float(offsets[-1]) + 1 / self.fps + .1,
            'absolute_tcp_poses', 'world', self.arm,
        )
        return chunk, end

    def provenance(self):
        return dict(self.metadata, selected_arm=self.arm, duration_seconds=self.duration,
                    playback='original action timestamps; no rescaling, recentering or looping',
                    other_arm='held at simulation startup',
                    grippers='original action opening [0, 1] (closed to open); calibrated URDF joints, '
                             'rate-limited offline simulation with native collision checks')
