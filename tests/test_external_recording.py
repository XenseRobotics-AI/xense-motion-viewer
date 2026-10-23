import json
import os
from pathlib import Path

import numpy as np
import pytest
from rerun.chunk import RrdReader

from robot_motion_workbench.adapters.reference import ReferenceRobotAdapter, description_from_urdf
from robot_motion_workbench.examples.reference import run_reference
from robot_motion_workbench.rerun_renderer import RerunRenderer, segment


@pytest.mark.parametrize('model', ['portable', 'rizon4'])
def test_real_recording_keeps_complete_reference_feedback_and_unknown_send(tmp_path, model):
    path = None
    if model == 'rizon4':
        if not os.environ.get('RMW_RIZON4_URDF'):
            pytest.skip('Set RMW_RIZON4_URDF for external licensed asset acceptance')
        path = Path(os.environ['RMW_RIZON4_URDF'])
    description = description_from_urdf(path)
    adapter = ReferenceRobotAdapter(description)
    output = tmp_path/'reference.rrd'
    result = run_reference(adapter, steps=24, save=output)
    assert result['display']['errors'] == result['display']['dropped_full'] == 0
    assert result['display_closed'] and not result['display']['alive']
    chunks = list(RrdReader(output).stream())
    documents = {}
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if 'TextDocument:text' in batch.schema.names:
            documents.setdefault(chunk.entity_path, []).extend(
                row[0] for row in batch.column('TextDocument:text').to_pylist() if row)
    events = [json.loads(text) for p in sorted(documents) if '/events/' in p for text in documents[p]]
    prediction = [e for e in events if e['stage'] == 'prediction']
    assert len(prediction) == 1
    assert np.shape(prediction[0]['chunk']['values']) == (12, 7)
    assert np.shape(prediction[0]['preview']['values']) == (12, 7)
    assert prediction[0]['timing']['source_time'] is None
    feedback = [e for e in events if e['stage'] == 'feedback']
    assert len(feedback) == 25
    assert all(e['origin'] == 'simulated' and e['timing']['source_clock'] == 'reference_simulation' for e in feedback)
    assert all('source_state_time' not in e['feedback'] for e in feedback)
    assert feedback[-1]['feedback']['joint_positions'] == result['final_joints']
    fallback = [e for e in events if e['stage'] == 'sent']
    assert len(fallback) == 12 and all(e['fallback'] and not e['available'] for e in fallback)
    panel = f'/policy/{segment(description.instance_id)}'
    assert panel + '/streams/sent' not in documents
    assert 'received **simulated** feedback' in documents[panel + '/status'][-1]
    paths = {c.entity_path for c in chunks}
    assert any('/preview/tool/tcp_path' in p for p in paths)
    assert any('/feedback_path/tool' in p for p in paths)
    for chunk in chunks:
        if chunk.entity_path.endswith('/preview/tool/tcp_path'):
            batch = chunk.to_record_batch()
            if 'LineStrips3D:strips' in batch.schema.names:
                # Full future is logged at availability, not at future measured times.
                assert set(batch.column('sandbox_elapsed').cast('int64').to_pylist()) == {0}
    tool = description.tool_frames['tool']
    fk = adapter.model.link_transforms(result['final_joints'])
    expected = fk[tool.parent_link] @ tool.link_from_tool
    entity = '/' + RerunRenderer.instance_path(description.instance_id) + '/tools/tool'
    matrices = []
    for chunk in chunks:
        batch = chunk.to_record_batch()
        if chunk.entity_path == entity and 'Transform3D:mat3x3' in batch.schema.names:
            matrices.extend(row[0] for row in batch.column('Transform3D:mat3x3').to_pylist() if row)
    assert len(matrices) == 25
    np.testing.assert_allclose(np.asarray(matrices[-1]).reshape(3, 3, order='F'), expected[:3, :3], atol=1e-6)
