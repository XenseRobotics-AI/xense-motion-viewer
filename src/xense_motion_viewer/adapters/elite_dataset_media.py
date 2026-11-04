"""Validated LeRobot-v3 image/task media for deterministic Elite RRDs."""
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from ..live_panels import ImageStream, LivePanels, TaskEvent, TaskSnapshot
from ..rerun_renderer import RerunRenderer

IMAGE_STREAMS = (
    ImageStream('head', 'Head RGB'),
    ImageStream('left_wrist', 'Left wrist RGB'),
    ImageStream('right_wrist', 'Right wrist RGB'),
    ImageStream('left_tactile_left', 'Left arm | Left tactile pad'),
    ImageStream('right_tactile_left', 'Right arm | Left tactile pad'),
    ImageStream('left_tactile_right', 'Left arm | Right tactile pad'),
    ImageStream('right_tactile_right', 'Right arm | Right tactile pad'),
)


@dataclass(frozen=True)
class EpisodeVideo:
    key: str
    path: Path
    source_from: float
    source_to: float

    def __post_init__(self):
        path = Path(self.path).resolve(strict=True)
        if (self.key not in {stream.key for stream in IMAGE_STREAMS}
                or not np.isfinite(self.source_from) or not np.isfinite(self.source_to)
                or self.source_from < 0 or self.source_to <= self.source_from):
            raise ValueError('Invalid Elite episode video mapping')
        object.__setattr__(self, 'path', path)
        object.__setattr__(self, 'source_from', float(self.source_from))
        object.__setattr__(self, 'source_to', float(self.source_to))


@dataclass(frozen=True)
class EpisodeMedia:
    root: Path
    episode_index: int
    frame_count: int
    fps: float
    task: str
    videos: Mapping[str, EpisodeVideo]
    source_info_sha256: str

    def __post_init__(self):
        videos = dict(self.videos)
        keys = tuple(stream.key for stream in IMAGE_STREAMS)
        if (type(self.episode_index) is not int or self.episode_index < 0
                or type(self.frame_count) is not int or self.frame_count < 1
                or not np.isfinite(self.fps) or self.fps <= 0
                or not isinstance(self.task, str) or tuple(videos) != keys):
            raise ValueError('Invalid Elite episode media')
        object.__setattr__(self, 'root', Path(self.root).resolve(strict=True))
        object.__setattr__(self, 'fps', float(self.fps))
        object.__setattr__(self, 'videos', MappingProxyType(videos))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_episode_media(root, episode_index):
    root = Path(root).resolve(strict=True)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if info.get('codebase_version') != 'v3.0' or info.get('robot_type') != 'bi_elite_cs66_rt':
        raise ValueError('Expected a LeRobot v3 bi_elite_cs66_rt dataset')
    keys = tuple(stream.key for stream in IMAGE_STREAMS)
    for key in keys:
        if info.get('features', {}).get(f'observation.images.{key}', {}).get('dtype') != 'video':
            raise ValueError(f'Missing Elite video feature: {key}')
    prefixes = [f'videos/observation.images.{key}' for key in keys]
    columns = ['episode_index', 'tasks', 'length', *(
        f'{prefix}/{field}' for prefix in prefixes
        for field in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp'))]
    matches = []
    for path in sorted((root / 'meta/episodes').glob('chunk-*/*.parquet')):
        matches.extend(pq.read_table(path, columns=columns,
            filters=[('episode_index', '=', episode_index)]).to_pylist())
    if len(matches) != 1:
        raise ValueError(f'Expected one metadata row for episode {episode_index}, found {len(matches)}')
    row = matches[0]
    tasks = row['tasks']
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
        raise ValueError('Expected exactly one task for the selected episode')
    videos = {}
    for key, prefix in zip(keys, prefixes, strict=True):
        relative = info['video_path'].format(video_key=f'observation.images.{key}',
            chunk_index=row[f'{prefix}/chunk_index'], file_index=row[f'{prefix}/file_index'])
        videos[key] = EpisodeVideo(key, root / relative,
            row[f'{prefix}/from_timestamp'], row[f'{prefix}/to_timestamp'])
    return EpisodeMedia(root, episode_index, row['length'], info['fps'], tasks[0], videos,
                        _sha256(info_path))


def create_episode_panels(recording, description, *, timeline='sandbox_elapsed'):
    motion = SimpleNamespace(timeline=timeline, description=description, robot=RerunRenderer)
    return LivePanels(recording, motion, IMAGE_STREAMS,
                      max_edge=640, jpeg_quality=85, stale_after=1)


def log_episode_media(recording, media, *, panels, source_dataset=None):
    panels.task(TaskSnapshot(0, generation=1, episode=media.episode_index,
        instruction=media.task, source='LeRobot v3 episode metadata', status='Offline dataset replay',
        events=(TaskEvent(1, 0, f'Episode {media.episode_index} replay started'),)))
    counts = {}
    hashes = {}
    for key, video in media.videos.items():
        counts[key] = len(panels.video(key, video.path, source_from=video.source_from,
                                      source_to=video.source_to, expected_frames=media.frame_count))
        hashes[key] = _sha256(video.path)
    duration = max(video.source_to - video.source_from for video in media.videos.values())
    panels.task_timeline(duration, force=True)
    provenance = {
        'source_dataset': source_dataset or str(media.root),
        'episode_index': media.episode_index,
        'frame_count': media.frame_count,
        'fps': media.fps,
        'source_info_sha256': media.source_info_sha256,
        'video_sha256': hashes,
        'video_reference_counts': counts,
        'video_semantics': 'Embedded source MP4; only the selected episode interval is referenced',
    }
    recording.set_time(panels.motion.timeline, duration=0)
    recording.log('live/source', panels.rr.TextDocument(
        json.dumps(provenance, indent=2), media_type='text/plain'), static=True)
    return provenance


__all__ = [
    'IMAGE_STREAMS',
    'EpisodeMedia',
    'EpisodeVideo',
    'create_episode_panels',
    'load_episode_media',
    'log_episode_media',
]
