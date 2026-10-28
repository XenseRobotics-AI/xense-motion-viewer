"""Review regressions using actual RRD rows and the bounded display worker."""
import json

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from rrd_reader import RrdReader, clear_times

from xense_motion_viewer import (CallbackRobotAdapter, ControlCapabilities,
    JointControl, JointTarget, PolicyChunk, PolicyInput, RobotDescription,
    RobotSession, RobotState, SampleTime, StepRef, StreamEvent, VisualizationWorker)
from xense_motion_viewer.policy_renderer import PolicyRerunRenderer


def rows(chunks, path, component):
    result = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if chunk.entity_path == '/' + path and component in batch.schema.names:
            times = batch.column('sandbox_elapsed').cast(pa.int64()).to_pylist()
            result.extend((t, row) for t, row in zip(times, batch.column(component).to_pylist()) if row is not None)
    return sorted(result, key=lambda row: row[0])


def documents(chunks, path):
    return [json.loads(value[0]) for _, value in rows(chunks, path, 'TextDocument:text')]


@pytest.fixture
def display(tmp_path, robot_urdf, names):
    path = tmp_path/'provenance.rrd'
    description = RobotDescription('review-fix', robot_urdf, names)
    recording = rr.RecordingStream('provenance-review-fix'); recording.save(path)
    renderer = PolicyRerunRenderer(recording, description)
    return renderer, path, description


def state(names, source_time, x):
    pose = np.eye(4); pose[0, 3] = x
    return RobotState(source_time, dict.fromkeys(names, x), tcp_poses={'tool': pose})


def test_unavailable_measured_feedback_retains_simulated_geometry_provenance(display, names):
    renderer, path, _ = display
    renderer.consume(StreamEvent('feedback', SampleTime(1, 30, 'sim-clock'),
                                  'simulated', state=state(names, 30, .1)))
    renderer.consume(StreamEvent('feedback', SampleTime(2), 'measured',
                                  reason='Hardware feedback disconnected'))
    renderer.close()
    chunks = list(RrdReader(path).stream())
    status = rows(chunks, f'{renderer.panel}/status', 'TextDocument:text')[-1][1][0]
    assert 'robot: received **simulated** feedback' in status
    assert 'host 1.000s; source 30.000s (sim-clock); age 1.000s (stale)' in status
    assert '| feedback | measured / unavailable |' in status
    assert 'Hardware feedback disconnected' in status
    rendered = documents(chunks, f'{renderer.panel}/rendered_feedback')
    assert len(rendered) == 1 and rendered[0]['origin'] == 'simulated'
    assert rendered[0]['timing']['received_at'] == 1
    assert documents(chunks, f'{renderer.panel}/streams/feedback')[-1]['origin'] == 'measured'
    link = renderer.robot.link_path('review-fix', 'link0')
    assert [t for t, _ in rows(chunks, link, 'Transform3D:translation')] == [1_000_000_000]


def test_available_origin_transitions_clear_tcp_history_without_changing_raw_stream(display, names):
    renderer, path, _ = display
    for host_time, provenance, x in ((1, 'simulated', .1), (2, 'measured', .8),
                                     (3, 'measured', .9), (4, 'simulated', .2)):
        renderer.consume(StreamEvent('feedback', SampleTime(host_time), provenance,
                                      state=state(names, host_time, x)))
    renderer.close()
    chunks = list(RrdReader(path).stream())
    paths = rows(chunks, f'{renderer.root}/feedback_path/tool', 'LineStrips3D:strips')
    assert [len(value[0]) for _, value in paths] == [1, 1, 2, 1]
    np.testing.assert_allclose([point[0] for point in paths[2][1][0]], [.8, .9])
    assert clear_times(chunks, f'{renderer.root}/feedback_path') == [2_000_000_000, 4_000_000_000]
    raw = documents(chunks, f'{renderer.panel}/streams/feedback')
    assert [e['origin'] for e in raw] == ['simulated', 'measured', 'measured', 'simulated']
    assert [e['feedback']['tcp_poses']['tool'][0][3] for e in raw] == [.1, .8, .9, .2]


def run_submissions(display, names, *, send_first=False, repeat=None, absent=False):
    renderer, path, description = display
    accepted = []
    caps = ControlCapabilities({n: JointControl() for n in names}, {'arm': names})
    worker = VisualizationWorker(renderer.consume, capacity=32, close_callback=renderer.close)
    def callback(command):
        accepted.append(command)
        if send_first and len(accepted) == 1:
            worker.publish(StreamEvent('sent', SampleTime(1.1, 43, 'send-clock'),
                'simulated', StepRef('A', 0), command=JointTarget('arm', dict.fromkeys(names, .51))))
        if absent:
            worker.publish(StreamEvent('sent', SampleTime(1.1), 'measured',
                StepRef('A', 0), reason='Transport reports no sample sent'))
    adapter = CallbackRobotAdapter(description, caps,
        read_state=lambda: state(names, 0, 0), submit_target=callback)
    policy = PolicyInput(RobotSession(adapter).activate('policy'), worker.publish)
    def offer(chunk_id, now):
        policy.accept(PolicyChunk(chunk_id, 'arm', names, [.1], [[.14]*len(names)],
                                  SampleTime(now), 4, 'absolute_joint_positions', 'joint'))
        policy.select(0, now=now)
    offer('A', 1)
    first = policy.execute_selected(now=1.1)
    if repeat:
        if repeat == 'new_chunk': offer('B', 2)
        policy.execute_selected(now=2.1)
    policy.close()
    assert worker.close(timeout=3) and worker.stats.errors == 0
    return renderer, list(RrdReader(path).stream()), first, accepted


def test_synchronous_send_callback_survives_fallback_with_real_worker(display, names):
    renderer, chunks, first, accepted = run_submissions(display, names, send_first=True)
    assert len(accepted) == 1 and first.receipt.status == 'submitted'
    status = rows(chunks, f'{renderer.panel}/status', 'TextDocument:text')[-1][1][0]
    assert '| sent | simulated / fresh | A[0] |' in status
    sent = documents(chunks, f'{renderer.panel}/streams/sent')
    assert len(sent) == 1 and sent[0]['command']['positions'][names[0]] == .51
    bookkeeping = documents(chunks, f'{renderer.panel}/bookkeeping/sent')
    assert len(bookkeeping) == 1 and bookkeeping[0]['fallback']
    assert bookkeeping[0]['submission_sequence'] == first.receipt.sequence


def test_no_send_evidence_stays_unavailable_without_invented_telemetry(display, names):
    renderer, chunks, _, _ = run_submissions(display, names)
    assert documents(chunks, f'{renderer.panel}/streams/sent') == []
    fallback = documents(chunks, f'{renderer.panel}/bookkeeping/sent')[-1]
    assert not fallback['available'] and fallback['command'] is None and fallback['fallback']
    status = rows(chunks, f'{renderer.panel}/status', 'TextDocument:text')[-1][1][0]
    assert '| sent | reference / unavailable | A[0] |' in status
    assert 'target acceptance supplies no send telemetry' in status


@pytest.mark.parametrize('repeat', ['new_chunk', 'same_step'])
def test_new_no_evidence_submission_is_separate_from_last_callback(display, names, repeat):
    renderer, chunks, _, accepted = run_submissions(display, names, send_first=True, repeat=repeat)
    assert len(accepted) == 2
    sent = documents(chunks, f'{renderer.panel}/streams/sent')
    assert len(sent) == 1 and sent[0]['ref'] == dict(chunk_id='A', step_index=0)
    fallbacks = documents(chunks, f'{renderer.panel}/bookkeeping/sent')
    assert [e['submission_sequence'] for e in fallbacks] == [1, 2]
    assert fallbacks[-1]['ref']['chunk_id'] == ('B' if repeat == 'new_chunk' else 'A')
    assert all(not e['available'] for e in fallbacks)
    status = rows(chunks, f'{renderer.panel}/status', 'TextDocument:text')[-1][1][0]
    assert 'Submission return #2 at host 2.100s' in status
    assert 'supplies no send evidence (fallback bookkeeping)' in status
    assert 'a chunk/step association alone does not identify a repeated submission' in status
    assert '| sent | simulated / stale | A[0] |' in status


def test_authoritative_absence_report_is_not_suppressed_by_fallback(display, names):
    renderer, chunks, _, _ = run_submissions(display, names, send_first=True, absent=True)
    sent = documents(chunks, f'{renderer.panel}/streams/sent')
    assert len(sent) == 2 and sent[0]['available'] and not sent[1]['available']
    assert sent[1]['fallback'] is False
    status = rows(chunks, f'{renderer.panel}/status', 'TextDocument:text')[-1][1][0]
    assert '| sent | measured / unavailable | A[0] |' in status
    assert 'Transport reports no sample sent' in status


def test_fallback_validation_rejects_authoritative_values_and_wrong_stage():
    with pytest.raises(ValueError):
        StreamEvent('feedback', SampleTime(0), 'simulated', reason='missing', fallback=True)
    with pytest.raises(ValueError):
        StreamEvent('sent', SampleTime(0), 'simulated', StepRef('A', 0),
                    command=JointTarget('arm', {'joint': .1}), fallback=True)
    with pytest.raises(ValueError):
        StreamEvent('sent', SampleTime(0), 'reference', StepRef('A', 0),
                    reason='missing', fallback=True, submission_sequence=0)
