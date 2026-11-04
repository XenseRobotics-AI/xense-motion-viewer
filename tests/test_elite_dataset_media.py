import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rerun as rr

from xense_motion_viewer.adapters.elite_dataset_media import (
    IMAGE_STREAMS,
    load_episode_media,
)
from xense_motion_viewer.live_panels import LivePanels
from xense_motion_viewer.policy_renderer import PolicyRerunRenderer


class _Recording:
    def __init__(self):
        self.blueprints = []
        self.columns = []

    def send_blueprint(self, blueprint):
        self.blueprints.append(blueprint)

    def set_time(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def send_columns(self, path, *, indexes, columns):
        self.columns.append((path, indexes, tuple(columns)))


@pytest.fixture
def video_panels(tmp_path, monkeypatch):
    # PTS returned by Rerun for the first nine frames of a real 30 Hz MP4.
    timestamps = np.array([0, 33333333, 66666667, 100000000, 133333333,
                           166666667, 200000000, 233333333, 266666667], dtype=np.int64)
    monkeypatch.setattr(rr.AssetVideo, 'read_frame_timestamps_nanos', lambda _: timestamps)
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'video')
    recording = _Recording()
    motion = SimpleNamespace(timeline='sandbox_elapsed',
        description=SimpleNamespace(instance_id='robot'),
        robot=SimpleNamespace(instance_path=lambda _: 'world/robots/robot'))
    panels = LivePanels(recording, motion, IMAGE_STREAMS,
                        max_edge=640, jpeg_quality=85, stale_after=1)
    return panels, path


@pytest.mark.parametrize('source_from, source_to, references, relative_nanos', [
    (0, 1 / 30, [0], [0]),
    (1 / 30, 4 / 30, [33333333, 66666667, 100000000], [0, 33333334, 66666667]),
    (1 / 30, 5 / 30, [33333333, 66666667, 100000000, 133333333],
     [0, 33333334, 66666667, 100000000]),
    (4 / 30, 7 / 30, [133333333, 166666667, 200000000], [0, 33333334, 66666667]),
    (5 / 30, 8 / 30, [166666667, 200000000, 233333333], [0, 33333333, 66666666]),
    (.04, .1, [66666667], [26666667]),
])
def test_video_intervals_preserve_source_frames_and_episode_time(
        video_panels, source_from, source_to, references, relative_nanos):
    panels, path = video_panels

    times = panels.video('head', path, source_from=source_from, source_to=source_to,
                        expected_frames=len(references))

    entity, indexes, columns = panels.recording.columns[0]
    assert entity == 'live/images/head'
    assert columns[0].as_arrow_array().to_pylist() == [[value] for value in references]
    assert indexes[0].as_arrow_array().cast(pa.int64()).to_pylist() == relative_nanos
    np.testing.assert_array_equal(times, np.array(relative_nanos) / 1e9)


@pytest.mark.parametrize('source_from, source_to, expected_frames, error', [
    (1 / 30, 4 / 30, 4, 'head has 3 video frames; expected 4'),
    (1., 2., 1, 'No video frames'),
    (-1., 1., 1, 'Video source interval'),
    (1., 1., 1, 'Video source interval'),
    (2., 1., 1, 'Video source interval'),
    (float('nan'), 1., 1, 'Video source interval'),
    (0., float('inf'), 1, 'Video source interval'),
])
def test_video_rejects_invalid_metadata_before_logging(
        video_panels, source_from, source_to, expected_frames, error):
    panels, path = video_panels

    with pytest.raises(ValueError, match=error):
        panels.video('head', path, source_from=source_from, source_to=source_to,
                     expected_frames=expected_frames)

    assert not panels.recording.columns


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
