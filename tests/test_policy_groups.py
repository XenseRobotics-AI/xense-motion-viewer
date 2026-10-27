from dataclasses import replace
import json
from threading import Event

import numpy as np
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from robot_motion_workbench import (JointTarget, PolicyChunk, RobotDescription, RobotState,
                                    SampleTime, StepRef, StreamEvent, VisualizationWorker)
from robot_motion_workbench.policy_renderer import PolicyRerunRenderer


def rows(chunks, path, component):
    return [value for chunk in chunks if chunk.entity_path == '/' + path
            for batch in [chunk.to_record_batch()] if component in batch.schema.names
            for value in batch.column(component).to_pylist() if value is not None]


def test_two_policy_groups_keep_both_paths_selections_and_telemetry(tmp_path, robot_urdf, names):
    recording = rr.RecordingStream('dual-policy')
    path = tmp_path / 'dual.rrd'
    recording.save(path)
    renderer = PolicyRerunRenderer(recording, RobotDescription('dual', robot_urdf, names),
                                   shared_view=True, telemetry=True, groups=('left', 'right'))
    chunks = {}
    for side, x in [('left', 1.), ('right', -1.)]:
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[:, :3, 3] = [[x, 0, 1], [x, .2, 1]]
        chunk = PolicyChunk(side + '-A', side, names, [0, 1], poses, SampleTime(0), 3,
                            'absolute_tcp_poses', 'world', side)
        chunks[side] = chunk
        renderer.consume(StreamEvent('prediction', chunk.timing, 'reference', chunk=chunk, group=side))
        renderer.consume(StreamEvent('selection', SampleTime(0), 'reference', StepRef(chunk.chunk_id, 1), group=side))
        renderer.consume(StreamEvent('planned', SampleTime(0), 'simulated', StepRef(chunk.chunk_id, 1),
                                     command=JointTarget(side, dict.fromkeys(names, x)), group=side))
    renderer.consume(StreamEvent('feedback', SampleTime(.1), 'simulated',
        state=RobotState(.1, dict.fromkeys(names, 0), tcp_poses={s: c.values[0] for s, c in chunks.items()})))
    newer = replace(chunks['left'], chunk_id='left-B', timing=SampleTime(.2))
    renderer.consume(StreamEvent('prediction', newer.timing, 'reference', chunk=newer, group='left'))
    assert renderer.robot.update_count == 1  # One full station update, not one per arm.
    renderer.close()
    reader = RrdReader(path)
    recorded = list(reader.stream())
    blueprints = [chunk for store in reader.blueprints() for chunk in reader.stream(store=store)]
    # The live scene must select simulation time and advance on new samples,
    # even if a previous Viewer recording was paused.
    assert rows(blueprints, 'time_panel', 'TimePanelBlueprint:timeline')
    assert all(value == ['sandbox_elapsed'] for value in
               rows(blueprints, 'time_panel', 'TimePanelBlueprint:timeline'))
    assert rows(blueprints, 'time_panel', 'TimePanelBlueprint:play_state')
    assert all(value == [3] for value in
               rows(blueprints, 'time_panel', 'TimePanelBlueprint:play_state'))
    for side, chunk in chunks.items():
        prefix = f'{renderer.root}/groups/{side}/chunks/{side}-A/prediction'
        np.testing.assert_allclose(rows(recorded, prefix + '/tcp_path', 'LineStrips3D:strips')[0][0], chunk.values[:, :3, 3])
        np.testing.assert_allclose(rows(recorded, prefix + '/selected', 'Points3D:positions')[0][0], chunk.values[1, :3, 3])
        assert rows(recorded, prefix + '/selected', 'Points3D:labels') == []
        assert rows(recorded, prefix + '/selected', 'Points3D:show_labels') == [[False]]
        # Selecting a target does not resend coordinates or claim submission.
        assert len(rows(recorded, prefix + '/tcp_path', 'LineStrips3D:strips')) == 1
        assert len(rows(recorded, prefix + '/steps', 'Points3D:colors')) == 1
        planned = rows(recorded, f'{renderer.panel}/groups/{side}/streams/planned', 'TextDocument:text')
        assert json.loads(planned[0][0])['command']['group'] == side
    assert rows(recorded, f'{renderer.root}/groups/left/chunks/left-A/prediction', 'Clear:is_recursive') == [[True]]
    assert rows(recorded, f'{renderer.root}/groups/right/chunks/right-A/prediction', 'Clear:is_recursive') == []
    status = rows(recorded, f'{renderer.panel}/status', 'TextDocument:text')[-1][0]
    assert '**left-B**' in status and '**right-A**' in status
    assert '### left' in status and '### right' in status
    for side in ('left', 'right'):
        assert rows(recorded, f'{renderer.root}/feedback_path/{side}', 'LineStrips3D:strips')


def test_display_coalescing_never_replaces_the_other_arms_planned_value():
    entered, release = Event(), Event()
    consumed = []
    def consume(event):
        if not consumed:
            entered.set()
            assert release.wait(3)
        consumed.append(event)
    worker = VisualizationWorker(consume, capacity=16, coalesce_feedback=True)
    try:
        worker.publish(StreamEvent('feedback', SampleTime(0), 'simulated', reason='No state yet'))
        assert entered.wait(2)
        def planned(side, value, time):
            return StreamEvent('planned', SampleTime(time), 'simulated',
                               command=JointTarget(side, {'joint': value}), group=side)
        old_left, right, left = planned('left', 1, .1), planned('right', 2, .2), planned('left', 3, .3)
        for event in (old_left, right, left):
            assert worker.publish(event)
        assert worker.stats.coalesced == 1
    finally:
        release.set()
        assert worker.close(timeout=3)
    assert consumed[1:] == [right, left]
    assert worker.stats.errors == 0


def test_group_mismatch_is_rejected(names):
    with pytest.raises(ValueError, match='group'):
        StreamEvent('planned', SampleTime(0), 'simulated',
                    command=JointTarget('left', dict.fromkeys(names, 0)), group='right')
