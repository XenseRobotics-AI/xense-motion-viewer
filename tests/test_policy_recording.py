from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.chunk import RrdReader

from xense_motion_viewer import (JointPreview, JointTarget, PolicyChunk,
    RobotDescription, RobotState, SampleTime, StepRef, StreamEvent, SubmissionReceipt,
    ToolFrame, replay_events)
from xense_motion_viewer.policy_renderer import PolicyRerunRenderer
from xense_motion_viewer.rerun_renderer import segment


def rows(chunks, path, component):
    values = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if chunk.entity_path == '/' + path and component in batch.schema.names:
            timestamps = batch.column('sandbox_elapsed').cast(pa.int64()).to_pylist()
            values += [(t, row_id, v) for t, row_id, v in zip(timestamps,
                batch.column('rerun.controls.RowId').to_pylist(), batch.column(component).to_pylist()) if v is not None]
    return [value for _, _, value in sorted(values)]


def times(chunks, path, component):
    result = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if chunk.entity_path == '/' + path and component in batch.schema.names:
            result += [t for t, value in zip(batch.column('sandbox_elapsed').cast(pa.int64()).to_pylist(),
                                             batch.column(component).to_pylist()) if value is not None]
    return result


def test_real_rrd_full_future_at_availability_distinct_streams_and_readonly_replay(tmp_path, robot_urdf, names):
    rr.set_strict_mode(True)
    description = RobotDescription('future robot', robot_urdf, names, tool_frames={'tool': ToolFrame('tool')})
    poses = np.tile(np.eye(4), (3, 1, 1)); poses[:, 0, 3] = [1, 2, 3]
    c = PolicyChunk('first/chunk', 'arm', names, [.5, 1, 2], poses,
                    SampleTime(0, 100, 'policy'), 3, 'absolute_tcp_poses', 'world', 'tool')
    ref = StepRef(c.chunk_id, 1)
    predicted = c.target(1)
    receipt = SubmissionReceipt(1, 'policy', predicted)
    planned, sent = [JointTarget('arm', dict.fromkeys(names, q)) for q in (.2, .3)]
    state = RobotState(50, dict.fromkeys(names, .4), tcp_poses={'tool': np.eye(4)})
    events = [StreamEvent('prediction', c.timing, 'reference', chunk=c),
              StreamEvent('selection', SampleTime(.1), 'reference', ref),
              StreamEvent('submitted', SampleTime(.1), 'reference', ref, receipt=receipt),
              StreamEvent('planned', SampleTime(.15), 'reference', ref, command=planned),
              StreamEvent('sent', SampleTime(.16, 7, 'sim_send'), 'simulated', ref, command=sent),
              StreamEvent('feedback', SampleTime(.2, 50, 'robot_sim'), 'simulated', ref, state=state)]
    # Same intended time .5 from a different availability epoch and chunk ID.
    newer = replace(c, chunk_id='second', timing=SampleTime(.25), time_offsets=[.25, .75, 1.75])
    events += [StreamEvent('superseded', SampleTime(.25), 'reference', ref),
               StreamEvent('prediction', newer.timing, 'reference', chunk=newer),
               StreamEvent('sent', SampleTime(.3), 'reference', StepRef('second', 0), reason='No send callback'),
               StreamEvent('expired', SampleTime(3), 'reference', StepRef('second', 0))]
    recording_path = tmp_path/'policy.rrd'
    recording = rr.RecordingStream('policy-readonly-replay'); recording.save(recording_path)
    renderer = PolicyRerunRenderer(recording, description)
    # Consumer has only description + immutable events; there is no adapter or
    # input lease to call even while it replays real command/feedback history.
    replay_events(events, renderer.consume)
    renderer.close()
    chunks = list(RrdReader(recording_path).stream())
    path = f'{renderer.root}/chunks/{segment(c.chunk_id)}/prediction'
    np.testing.assert_allclose(rows(chunks, f'{path}/tcp_path', 'LineStrips3D:strips')[0],
                               np.stack((poses[:-1, :3, 3], poses[1:, :3, 3]), axis=1))
    assert times(chunks, f'{path}/tcp_path', 'LineStrips3D:strips') == [0]
    assert times(chunks, f'{path}/steps', 'Points3D:positions') == [0]
    radii = rows(chunks, f'{path}/steps', 'Points3D:radii')[-1]
    assert radii[0] == radii[2] > radii[1] == 0
    assert times(chunks, f'{path}/axes/2', 'Transform3D:translation') == [0]
    assert rows(chunks, path, 'Clear:is_recursive') == [[True]]
    path2 = f'{renderer.root}/chunks/second/prediction'
    assert times(chunks, f'{path2}/tcp_path', 'LineStrips3D:strips') == [250_000_000]
    assert rows(chunks, path2, 'Clear:is_recursive') == [[True]]
    link = renderer.robot.link_path(description.instance_id, 'link0')
    assert times(chunks, link, 'Transform3D:translation') == [200_000_000]
    # Source 50 seconds and intended future times never become measured times.
    metadata = {stage: [json.loads(row[0]) for row in rows(chunks,
                f'{renderer.panel}/streams/{stage}', 'TextDocument:text')]
                for stage in ('prediction', 'submitted', 'planned', 'sent', 'feedback')}
    assert metadata['prediction'][0]['chunk']['time_offsets'] == [.5, 1, 2]
    assert metadata['prediction'][1]['chunk']['intended_host_times'][0] == .5
    assert metadata['submitted'][0]['receipt']['command']['world_from_tcp'][0][3] == 2
    assert metadata['planned'][0]['command']['positions'][names[0]] == .2
    assert metadata['sent'][0]['command']['positions'][names[0]] == .3
    assert metadata['sent'][1]['available'] is False and metadata['sent'][1]['command'] is None
    assert metadata['feedback'][0]['feedback']['joint_positions'][names[0]] == .4
    assert metadata['feedback'][0]['timing']['source_time'] == 50
    statuses = rows(chunks, f'{renderer.panel}/status', 'TextDocument:text')
    assert 'unavailable' in statuses[-1][0] and 'stale' in statuses[-1][0]


def test_joint_preview_ghosts_are_references_at_chunk_time(tmp_path, robot_urdf, names):
    description = RobotDescription('ghost', robot_urdf, names, tool_frames={'tool': ToolFrame('tool')})
    c = PolicyChunk('joints', 'arm', names, [.1, .2], np.full((2, 7), .1),
                    SampleTime(.5), 2, 'absolute_joint_positions', 'joint')
    preview = JointPreview(names, c.values, 'SDK full-joint preview')
    recording = rr.RecordingStream('preview'); path = tmp_path/'preview.rrd'; recording.save(path)
    renderer = PolicyRerunRenderer(recording, description)
    renderer.consume(StreamEvent('prediction', c.timing, 'reference', chunk=c, preview=preview))
    renderer.close()
    chunks = list(RrdReader(path).stream())
    ghost = f'{renderer.root}/chunks/joints/prediction/ghosts/1'
    assert len(rows(chunks, ghost, 'LineStrips3D:strips')[0]) == len(renderer.model.ordered_joints)
    assert times(chunks, ghost, 'LineStrips3D:strips') == [500_000_000]
    joint = f'{renderer.panel}/joint_reference/{segment(names[0])}'
    np.testing.assert_allclose(rows(chunks, joint, 'LineStrips2D:strips')[0][0], [[.1, .1], [.2, .1]])


def test_event_history_keeps_bounded_entities_and_same_time_event_order(tmp_path, robot_urdf, names):
    path = tmp_path / 'event-history.rrd'
    recording = rr.RecordingStream('event-history'); recording.save(path)
    renderer = PolicyRerunRenderer(recording, RobotDescription('history', robot_urdf, names))
    renderer.consume_batch([StreamEvent('planned', SampleTime(0), 'simulated',
        command=JointTarget('arm', dict.fromkeys(names, index / 100))) for index in range(100)])
    renderer.close()
    chunks = list(RrdReader(path).stream())
    event_paths = {chunk.entity_path for chunk in chunks if '/events/' in chunk.entity_path}
    assert event_paths == {'/' + renderer.panel + '/events/log'}
    history = [json.loads(row[0]) for row in rows(chunks, renderer.panel + '/events/log', 'TextDocument:text')]
    history.sort(key=lambda event: event['event_sequence'])
    assert [event['event_sequence'] for event in history] == list(range(1, 101))
    assert [event['command']['positions'][names[0]] for event in history] == [i / 100 for i in range(100)]


def test_batched_selection_hides_submitted_geometry_and_updates_latest_marker(tmp_path, robot_urdf, names):
    path = tmp_path / 'progress.rrd'
    recording = rr.RecordingStream('progress'); recording.save(path)
    renderer = PolicyRerunRenderer(recording, RobotDescription('progress', robot_urdf, names), shared_view=True)
    poses = np.tile(np.eye(4), (4, 1, 1)); poses[:, 0, 3] = np.arange(4)
    chunk = PolicyChunk('A', 'arm', names, [0, 1, 2, 3], poses,
                        SampleTime(0), 10, 'absolute_tcp_poses', 'world', 'tool')
    events = [StreamEvent('prediction', chunk.timing, 'reference', chunk=chunk)]
    for i in (0, 1, 3):
        ref = StepRef('A', i)
        events.append(StreamEvent('selection', SampleTime(i), 'reference', ref))
        # Selecting step 3 and submitting its gripper must not claim the TCP
        # was submitted, or mark skipped step 2 as completed.
        target = chunk.target(i) if i < 3 else JointTarget('gripper', {'finger': .1})
        events.append(StreamEvent('submitted', SampleTime(i), 'reference', ref,
                                  receipt=SubmissionReceipt(i + 1, 'policy', target)))
    renderer.consume_batch(events)
    renderer.close()
    recorded = list(RrdReader(path).stream())
    prefix = f'{renderer.root}/chunks/A/prediction'
    assert len(rows(recorded, prefix + '/tcp_path', 'LineStrips3D:strips')) == 1
    assert len(rows(recorded, prefix + '/steps', 'Points3D:positions')) == 1
    radii = rows(recorded, prefix + '/steps', 'Points3D:radii')
    assert len(radii) == 2
    np.testing.assert_allclose(radii[-1], [0, 0, .006, .006])
    line_radii = rows(recorded, prefix + '/tcp_path', 'LineStrips3D:radii')[-1]
    np.testing.assert_allclose(line_radii, [0, .006, .006])
    np.testing.assert_allclose(rows(recorded, prefix + '/selected', 'Points3D:positions'), [[[3, 0, 0]]])
    assert len(rows(recorded, renderer.panel + '/events/log', 'TextDocument:text')) == len(events)


def test_batched_feedback_keeps_all_samples_and_bounded_incremental_trail(tmp_path, robot_urdf, names):
    path = tmp_path / 'trail.rrd'
    recording = rr.RecordingStream('trail'); recording.save(path)
    renderer = PolicyRerunRenderer(recording, RobotDescription('trail', robot_urdf, names),
                                   shared_view=True, compact=True)
    count = 2240
    for start in range(0, count, 64):
        events = []
        for i in range(start, start + 64):
            pose = np.eye(4); pose[0, 3] = i
            state = RobotState(i, dict.fromkeys(names, 0), tcp_poses={'tool': pose})
            events.append(StreamEvent('feedback', SampleTime(i / 30), 'simulated', state=state))
        renderer.consume_batch(events)
    assert renderer.robot.update_count == count // 64
    renderer.close()
    recorded = list(RrdReader(path).stream())
    raw = rows(recorded, renderer.panel + '/streams/feedback', 'TextDocument:text')
    assert len(raw) == count  # Batching reduces draws, never raw evidence.
    prefix = '/' + renderer.root + '/feedback_path/tool'
    blocks, total_points = {}, 0
    for chunk in recorded:
        batch = chunk.to_record_batch()
        if chunk.entity_path.startswith(prefix) and 'LineStrips3D:strips' in batch.schema.names:
            for time, value in zip(batch.column('sandbox_elapsed').cast(pa.int64()).to_pylist(),
                                   batch.column('LineStrips3D:strips').to_pylist()):
                if value is not None:
                    assert len(value[0]) <= 64
                    total_points += len(value[0])
                    if time >= blocks.get(chunk.entity_path, (-1,))[0]:
                        blocks[chunk.entity_path] = time, value[0]
    assert len(blocks) == 32 and total_points < 4 * count
    edges = {(int(a[0]), int(b[0])) for _, points in blocks.values() for a, b in zip(points, points[1:])}
    first = min(a for a, _ in edges)
    assert first > 0 and edges == {(i, i + 1) for i in range(first, count - 1)}
    endpoint = rows(recorded, renderer.root + '/feedback_path/tool/current', 'Points3D:positions')[-1]
    np.testing.assert_allclose(endpoint, [[count - 1, 0, 0]])
