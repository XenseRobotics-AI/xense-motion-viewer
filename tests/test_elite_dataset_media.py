import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from xense_motion_viewer.adapters.elite_dataset_media import (
    IMAGE_STREAMS,
    load_episode_media,
)
from xense_motion_viewer.live_panels import LivePanels
from xense_motion_viewer.policy_renderer import PolicyRerunRenderer


class _Recording:
    def __init__(self):
        self.blueprints = []

    def send_blueprint(self, blueprint):
        self.blueprints.append(blueprint)

    def set_time(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def test_combined_layout_can_be_reasserted_after_motion_changes_blueprint():
    recording = _Recording()
    motion = SimpleNamespace(
        timeline='sandbox_elapsed',
        description=SimpleNamespace(instance_id='robot'),
        robot=SimpleNamespace(instance_path=lambda _: 'world/robots/robot'),
    )
    panels = LivePanels(recording, motion, IMAGE_STREAMS,
                        max_edge=640, jpeg_quality=85, stale_after=1)
    assert len(recording.blueprints) == 1

    panels.blueprint()

    assert len(recording.blueprints) == 2


def test_elite_image_rows_keep_head_large_then_pair_wrists_and_tactile_sides():
    recording = _Recording()
    motion = SimpleNamespace(
        timeline='sandbox_elapsed',
        description=SimpleNamespace(instance_id='robot'),
        robot=SimpleNamespace(instance_path=lambda _: 'world/robots/robot'),
    )
    panels = LivePanels(recording, motion, IMAGE_STREAMS,
                        max_edge=640, jpeg_quality=85, stale_after=1)

    assert tuple(tuple(stream.key for stream in row) for row in panels._image_rows()) == (
        ('head',),
        ('left_wrist', 'right_wrist'),
        ('left_tactile_left', 'right_tactile_left'),
        ('left_tactile_right', 'right_tactile_right'),
    )


def test_policy_renderer_delegates_dynamic_layout_changes_to_combined_owner():
    renderer = PolicyRerunRenderer.__new__(PolicyRerunRenderer)
    renderer._blueprint_owner = None
    calls = []

    renderer.set_blueprint_owner(lambda: calls.append('combined'))
    renderer.blueprint()

    assert calls == ['combined']


def test_load_episode_media_filters_shared_metadata_and_resolves_all_cameras(tmp_path):
    keys = tuple(stream.key for stream in IMAGE_STREAMS)
    info = {
        'codebase_version': 'v3.0',
        'robot_type': 'bi_elite_cs66_rt',
        'fps': 30,
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'features': {f'observation.images.{key}': {'dtype': 'video'} for key in keys},
    }
    (tmp_path / 'meta/episodes/chunk-000').mkdir(parents=True)
    (tmp_path / 'meta/info.json').write_text(json.dumps(info))
    rows = []
    for episode, file_index, length in ((1, 1, 4), (2, 1, 5)):
        row = {'episode_index': episode, 'tasks': [f'task-{episode}'], 'length': length}
        for key in keys:
            prefix = f'videos/observation.images.{key}'
            row.update({f'{prefix}/chunk_index': 0, f'{prefix}/file_index': file_index,
                        f'{prefix}/from_timestamp': 0., f'{prefix}/to_timestamp': length / 30})
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'meta/episodes/chunk-000/file-001.parquet')
    for key in keys:
        path = tmp_path / f'videos/observation.images.{key}/chunk-000/file-001.mp4'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'video')

    media = load_episode_media(tmp_path, 1)

    assert media.episode_index == 1
    assert media.frame_count == 4
    assert media.task == 'task-1'
    assert tuple(media.videos) == keys
    assert all(video.path.name == 'file-001.mp4' for video in media.videos.values())
