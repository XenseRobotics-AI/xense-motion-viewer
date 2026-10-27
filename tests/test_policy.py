from dataclasses import replace
from threading import Event, Thread

import numpy as np
import pytest

from robot_motion_workbench import (
    CallbackRobotAdapter, ControlCapabilities, JointControl, JointPreview,
    JointTarget, PolicyChunk, PolicyInput, RobotDescription, RobotSession,
    RobotState, SampleTime, StepRef, StreamEvent, TcpControl, VisualizationWorker,
    replay_events,
)
from robot_motion_workbench.policy_renderer import event_data


@pytest.fixture
def host(robot_urdf, names):
    accepted, events = [], []
    description = RobotDescription('policy-test', robot_urdf, names)
    caps = ControlCapabilities({n: JointControl(lower=-2, upper=2) for n in names},
                               {'arm': names}, {'tool': TcpControl('arm')})
    adapter = CallbackRobotAdapter(description, caps,
        read_state=lambda: RobotState(0, dict.fromkeys(names, 0)), submit_target=accepted.append)
    session = RobotSession(adapter)
    policy = PolicyInput(session.activate('policy'), events.append)
    return policy, session, accepted, events


def chunk(names, **kwargs):
    data = dict(chunk_id='chunk-a', group='arm', joint_names=names,
        time_offsets=[0.2, 0.5, 1.0], values=np.tile([0.1, 0.2, 0.3], (len(names), 1)).T,
        timing=SampleTime(0, 40, 'policy_clock'), expires_at=2,
        kind='absolute_joint_positions', frame='joint')
    data.update(kwargs)
    return PolicyChunk(**data)


@pytest.mark.parametrize('change', [
    {'joint_names': ('same', 'same')}, {'joint_names': ()},
    {'time_offsets': [0.5, 0.2, 1]}, {'time_offsets': [0, 0, 1]},
    {'time_offsets': [-1, 0, 1]}, {'time_offsets': [0, np.nan, 1]},
    {'values': [[0]]}, {'values': np.full((3, 7), np.inf)},
    {'kind': 'delta_joint_positions'}, {'frame': 'camera'},
    {'expires_at': 0}, {'origin': 'measured'},
])
def test_chunk_rejects_ambiguous_or_invalid_actions(names, change):
    with pytest.raises((ValueError, TypeError)):
        chunk(names, **change)


def test_tcp_json_validation_and_immutable_history(names):
    poses = np.tile(np.eye(4), (3, 1, 1)); poses[:, 0, 3] = [.1, .2, .3]
    offsets = np.array([0.2, 0.5, 1])
    c = chunk(names, values=poses, time_offsets=offsets, kind='absolute_tcp_poses', frame='world', tcp='tool')
    event = StreamEvent('prediction', c.timing, 'reference', chunk=c)
    poses[:] = 0; offsets[:] = 100
    np.testing.assert_array_equal(c.values[:, 0, 3], [.1, .2, .3])
    np.testing.assert_array_equal(c.time_offsets, [.2, .5, 1])
    with pytest.raises(ValueError):
        c.values.setflags(write=True)
    data = event_data(event)['chunk']; data.pop('intended_host_times')
    parsed = PolicyChunk.from_dict(data)
    np.testing.assert_array_equal(parsed.values, c.values)
    for mutate in ('rotation', 'last_row', 'nan', 'unknown_field', 'unknown_clock'):
        invalid = event_data(event)['chunk']; invalid.pop('intended_host_times')
        if mutate == 'rotation': invalid['values'][1][0][0] = 2
        if mutate == 'last_row': invalid['values'][2][3][0] = 1
        if mutate == 'nan': invalid['values'][0][0][3] = float('nan')
        if mutate == 'unknown_field': invalid['normalized'] = True
        if mutate == 'unknown_clock': invalid['timing']['source_clock'] = None
        with pytest.raises((ValueError, TypeError)):
            PolicyChunk.from_dict(invalid)


def test_all_steps_validate_names_groups_bounds_and_tcp(host, names):
    policy, _, accepted, _ = host
    invalid = [chunk(names, joint_names=(*names[:-1], 'unknown')),
               chunk(names, group='other'), chunk(names, values=np.full((3, 7), 3)),
               chunk(names, values=np.tile(np.eye(4), (3, 1, 1)),
                     kind='absolute_tcp_poses', frame='world', tcp='unknown')]
    for c in invalid:
        with pytest.raises(ValueError): policy.accept(c)
    assert accepted == [] and policy.active_chunk is None
    # Deliberately reverse input order: submission follows names, not indices.
    c = chunk(tuple(reversed(names)), values=np.arange(21).reshape(3, 7) / 100)
    policy.accept(c)
    for index in (-1, 3, True, 1.5):
        with pytest.raises(ValueError): policy.select(index, now=0)
    policy.select(2, now=0)
    result = policy.execute_selected(now=0)
    assert tuple(result.receipt.command.positions) == names
    assert result.receipt.command.positions[names[0]] == .20


def test_replacement_expiry_and_selected_index_are_explicit(host, names):
    policy, session, accepted, events = host
    first = chunk(names, time_offsets=[10, 11, 12], expires_at=20)
    policy.accept(first); policy.select(2, now=0)
    assert accepted == []
    policy.execute_selected(now=0)
    # Two generation times can have exactly the same intended execution time.
    second = chunk(names, chunk_id='chunk-b', timing=SampleTime(1), time_offsets=[9, 10, 11])
    policy.accept(second)
    assert first.timing.received_at + first.time_offsets[0] == second.timing.received_at + second.time_offsets[0]
    superseded = [e for e in events if e.stage == 'superseded'][0]
    assert superseded.ref == StepRef('chunk-a', 2)
    with pytest.raises(RuntimeError): policy.execute_selected(now=1)
    policy.select(1, now=1)
    submission = policy.execute_selected(now=1)
    assert submission.ref == StepRef('chunk-b', 1)
    assert len(accepted) == 2 and session.last_receipt.command.positions[names[0]] == .2
    policy.expire(2)
    assert policy.active_chunk is None and policy.selected is None
    with pytest.raises(RuntimeError): policy.execute_selected(now=2)
    assert events[-1].stage == 'expired' and events[-1].ref == submission.ref
    assert len(accepted) == 2  # expiry does not cancel or command
    policy.close(); policy.close()
    with pytest.raises(RuntimeError): policy.select(0, now=3)


def test_rejection_does_not_mutate_lifecycle_clock(host, names):
    policy, _, accepted, _ = host
    original = chunk(names)
    policy.accept(original); policy.select(0, now=0)
    invalid = chunk(names, chunk_id='invalid', group='unknown', timing=SampleTime(100), expires_at=101)
    with pytest.raises(ValueError): policy.accept(invalid)
    with pytest.raises(ValueError):
        policy.accept(replace(original, timing=SampleTime(100), expires_at=101))
    with pytest.raises(ValueError):
        policy.accept(replace(original, chunk_id='bad-preview', timing=SampleTime(100), expires_at=101),
                      preview=JointPreview(names[:-1], np.zeros((3, 6)), 'incomplete'))
    assert policy.active_chunk is original and policy.selected == StepRef('chunk-a', 0)
    policy.execute_selected(now=.1)
    assert len(accepted) == 1


def test_lease_takeover_owner_thread_and_display_failure_are_separate(host, names):
    policy, session, accepted, _ = host
    def broken(event): raise RuntimeError('display failed')
    policy.publish = broken
    policy.accept(chunk(names)); policy.select(1, now=0)
    result = policy.execute_selected(now=0)
    assert result.receipt.status == 'submitted' and len(accepted) == 1
    assert policy.display_errors == 4 and 'display failed' in policy.last_display_error
    errors = []
    def other_thread():
        try: policy.execute_selected(now=0)
        except Exception as exc: errors.append(exc)
    thread = Thread(target=other_thread); thread.start(); thread.join()
    assert isinstance(errors[0], RuntimeError) and len(accepted) == 1
    replacement = session.activate('qt', replace=True)
    with pytest.raises(RuntimeError): policy.execute_selected(now=0)
    policy.close()
    assert replacement.active


def test_distinct_prediction_submission_planned_sent_feedback_and_replay(host, names):
    policy, _, accepted, events = host
    values = np.full((3, 7), .1)
    policy.accept(chunk(names, values=values)); policy.select(1, now=0)
    submitted = policy.execute_selected(now=0)
    values[:] = 9
    assert events[-1].stage == 'sent' and not events[-1].available
    assert events[-1].command is None and events[-1].origin == 'reference'
    planned = JointTarget('arm', dict.fromkeys(names, .2))
    sent = JointTarget('arm', dict.fromkeys(names, .3))
    source_positions = dict.fromkeys(names, .4)
    feedback = RobotState(5, source_positions)
    source_positions[names[0]] = 9
    events += [StreamEvent('planned', SampleTime(.1, 7, 'planner'), 'reference', submitted.ref, command=planned),
               StreamEvent('sent', SampleTime(.2, 8, 'simulator_send'), 'simulated', submitted.ref, command=sent),
               StreamEvent('feedback', SampleTime(.3), 'simulated', state=feedback)]
    data = []
    replay_events(events, lambda e: data.append(event_data(e)))
    assert len(accepted) == 1  # neither playback nor display calls submit
    assert data[0]['chunk']['values'][0][0] == .1
    assert data[2]['receipt']['command']['positions'][names[0]] == .1
    assert data[-3]['command']['positions'][names[0]] == .2
    assert data[-2]['command']['positions'][names[0]] == .3
    assert data[-1]['feedback']['joint_positions'][names[0]] == .4
    assert data[-1]['ref'] is None and data[-1]['timing']['source_time'] is None
    assert events[-1].timing.freshness(.4, .2) == 'fresh'
    assert events[-1].timing.freshness(.6, .2) == 'stale'
    with pytest.raises(ValueError):
        StreamEvent('sent', SampleTime(.1), 'reference', command=sent)


def test_slow_display_cannot_stall_submission_and_queue_is_bounded(host, names):
    policy, _, accepted, _ = host
    entered, release = Event(), Event()
    consumed, finished = [], []
    def slow(event):
        entered.set(); release.wait(5); consumed.append(event)
    worker = VisualizationWorker(slow, capacity=2, close_callback=lambda: finished.append(True))
    try:
        policy.publish = worker.publish
        policy.accept(chunk(names))
        assert entered.wait(1)
        # Consumer remains blocked throughout a real session submission and 100
        # publications. No timing threshold or renderer speed assumption needed.
        policy.select(2, now=0)
        receipt = policy.execute_selected(now=0)
        for _ in range(100): worker.publish('extra')
        assert receipt.receipt.status == 'submitted' and len(accepted) == 1
        stats = worker.stats
        assert stats.pending == 2 and stats.dropped_full >= 100 and not release.is_set()
        assert not worker.close(timeout=.01)
        assert not worker.publish('closed') and worker.stats.rejected_closed == 1
    finally:
        release.set()
        assert worker.close(timeout=2)
    assert len(consumed) == 3 and finished == [True]
    assert worker.stats.consumed == worker.stats.accepted == 3


def test_consumer_error_and_close_discard_are_observable():
    entered, release = Event(), Event()
    def consumer(event):
        entered.set(); release.wait(2); raise ValueError('bad display')
    worker = VisualizationWorker(consumer, capacity=2)
    worker.publish(1); assert entered.wait(1)
    worker.publish(2); worker.publish(3)
    assert not worker.close(drain=False, timeout=.01)
    release.set(); assert worker.close(timeout=2)
    assert worker.stats.errors == 1 and worker.stats.consumed == 1
    assert worker.stats.discarded_close == 2 and worker.stats.dropped_full == 0
    assert worker.stats.pending == 0 and 'bad display' in worker.stats.last_error


def test_display_batches_preserve_order_and_reject_whole_frames_when_full():
    entered, release = Event(), Event()
    batches = []

    def consume(events):
        if not batches:
            entered.set()
            assert release.wait(3)
        batches.append(events)

    worker = VisualizationWorker(consume, capacity=4, batch=True)
    try:
        assert worker.publish_many([0, 1])
        assert entered.wait(2)
        assert worker.publish_many([2, 3, 4])
        assert not worker.publish_many([5, 6])
        assert worker.stats.pending == 3 and worker.stats.dropped_full == 2
        assert worker.publish_many([7])
    finally:
        release.set()
        assert worker.close(timeout=3)
    assert batches == [(0, 1), (2, 3, 4, 7)]
    assert worker.stats.consumed == worker.stats.accepted == 6
    assert worker.stats.errors == worker.stats.coalesced == 0


def test_continuous_telemetry_cannot_starve_pending_feedback():
    entered, release, done = Event(), Event(), Event()
    consumed = []
    sent_count, clock = 0, 4

    def sent(t):
        return StreamEvent('sent', SampleTime(t), 'reference', reason='No transport sample')

    def feedback(t):
        return StreamEvent('feedback', SampleTime(t), 'simulated', state=RobotState(t, {'q': t}))

    def consume(event):
        nonlocal sent_count, clock
        if not consumed:
            entered.set()
            assert release.wait(3)
        consumed.append(event)
        if event.stage == 'sent':
            sent_count += 1
            if sent_count < 20:
                # Deterministic continuous producer: every consumed telemetry
                # event brings more telemetry and a newer robot pose.
                clock += 2
                worker.publish(sent(clock - 1))
                worker.publish(feedback(clock))
            else:
                done.set()

    worker = VisualizationWorker(consume, capacity=16, coalesce_feedback=True)
    try:
        worker.publish(sent(0))
        assert entered.wait(2)
        for event in (feedback(1), sent(2), feedback(3), sent(4)):
            worker.publish(event)
        release.set()
        assert done.wait(3)
    finally:
        release.set()
        assert worker.close(timeout=3)
    assert next(i for i, event in enumerate(consumed) if event.stage == 'feedback') < 4
    assert worker.stats.errors == worker.stats.dropped_full == 0
    times = [event.timing.received_at for event in consumed]
    assert times == sorted(times)
