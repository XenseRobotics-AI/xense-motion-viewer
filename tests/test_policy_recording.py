from dataclasses import replace
import json

import numpy as np
import pyarrow as pa
import rerun as rr
from rerun.chunk import RrdReader

from robot_motion_workbench import (JointPreview, JointTarget, PolicyChunk,
    RobotDescription, RobotState, SampleTime, StepRef, StreamEvent, SubmissionReceipt,
    ToolFrame, replay_events)
from robot_motion_workbench.policy_renderer import PolicyRerunRenderer
from robot_motion_workbench.rerun_renderer import segment


def rows(chunks, path, component):
    values = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if chunk.entity_path == '/' + path and component in batch.schema.names:
            values += [v for v in batch.column(component).to_pylist() if v is not None]
    return values


def times(chunks, path, component):
    result = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if chunk.entity_path == '/' + path and component in batch.schema.names:
            result += batch.column('sandbox_elapsed').cast(pa.int64()).to_pylist()
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
    np.testing.assert_allclose(rows(chunks, f'{path}/tcp_path', 'LineStrips3D:strips')[0][0], poses[:, :3, 3])
    assert times(chunks, f'{path}/tcp_path', 'LineStrips3D:strips') == [0, 100000000]
    np.testing.assert_allclose(rows(chunks, f'{path}/tcp_path', 'LineStrips3D:strips')[1][0], poses[1:, :3, 3])
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


def test_rrd_replay_cli_only_launches_viewer(tmp_path, monkeypatch):
    from robot_motion_workbench.adapters import elite_policy
    path = tmp_path/'recording.rrd'; path.write_bytes(b'fake viewer path')
    calls = []
    class Result:
        returncode = 0
    def viewer(args, **kwargs):
        calls.append(args)
        return Result()
    monkeypatch.setattr(elite_policy.subprocess, 'run', viewer)
    assert elite_policy.replay_main([str(path)]) == 0
    assert calls[0][1:] == ['-m', 'rerun', str(path)]
