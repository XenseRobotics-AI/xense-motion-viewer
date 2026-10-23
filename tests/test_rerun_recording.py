import subprocess
import sys

import numpy as np
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from robot_motion_workbench import RobotDescription, RobotState, ToolFrame
from robot_motion_workbench.rerun_renderer import RerunRenderer, segment


def column_rows(chunks, path, component):
    values = []
    for chunk in chunks:
        if chunk.entity_path != '/' + path:
            continue
        batch = chunk.to_record_batch()
        index = batch.schema.get_field_index(component)
        if index >= 0:
            values.extend(x for x in batch.column(index).to_pylist() if x is not None)
    return values


def test_real_sdk_namespaces_transforms_and_static_geometry(tmp_path, robot_urdf, names):
    rr.set_strict_mode(True)
    path = tmp_path/'generic.rrd'
    rec = rr.RecordingStream('rmw-test')
    rec.save(path)
    renderer = RerunRenderer(rec)
    base = np.eye(4); base[0, 3] = 5
    ids = ['robot/a', 'robot%2Fa']
    models = [renderer.add_robot(RobotDescription(i, robot_urdf, names, base,
               tool_frames={'tcp': ToolFrame('tool')})) for i in ids]
    assert renderer.geometry_log_count == 4
    with pytest.raises(ValueError, match='Duplicate'):
        renderer.add_robot(models[0].description)
    q = dict.fromkeys(names, 0)
    q[names[0]], q[names[1]] = np.pi/2, 2
    actual = np.eye(4); actual[:3, 3] = [4, 5, 6]
    target = np.eye(4); target[:3, 3] = [7, 8, 9]
    for t in range(3):
        renderer.update(ids[0], RobotState(t, q, tcp_poses={'custom/tip': actual}, targets={'custom/tip': target}))
        renderer.update(ids[1], RobotState(t, q, base_transform=np.eye(4)))
    renderer.update(ids[0], RobotState(3, q))  # omitted dynamic frames get cleared
    renderer.set_collision_visible(True)
    renderer.set_collision_visible(False)
    assert renderer.geometry_log_count == 4
    with pytest.raises(ValueError, match='Time moved'):
        renderer.update(ids[0], RobotState(1, q))
    with pytest.raises(ValueError, match='exactly match'):
        renderer.update(ids[0], RobotState(4, {}))
    rec.flush(); rec.disconnect()
    reader = RrdReader(path)
    chunks = list(reader.stream())
    queries = [row for blueprint in reader.blueprints() for chunk in reader.stream(store=blueprint)
               for batch in [chunk.to_record_batch()] if 'ViewContents:query' in batch.schema.names
               for row in batch.column(batch.schema.get_field_index('ViewContents:query')).to_pylist()]
    assert any(all(f'- /{renderer.instance_path(i)}/collision/**' in query for i in ids) for query in queries)
    link = renderer.link_path(ids[0], 'link1')
    values = column_rows(chunks, link, 'Transform3D:translation')
    assert len(values) == 4
    np.testing.assert_allclose(values[0][0], [6, 0, 2], atol=1e-6)
    rotation = column_rows(chunks, link, 'Transform3D:mat3x3')[0][0]
    np.testing.assert_allclose(np.array(rotation).reshape(3,3,order='F'), [[0,0,1],[1,0,0],[0,1,0]], atol=1e-6)
    link2 = renderer.link_path(ids[1], 'link1')
    np.testing.assert_allclose(column_rows(chunks, link2, 'Transform3D:translation')[0][0], [1,0,2], atol=1e-6)
    root = renderer.instance_path(ids[0])
    for group, expected in [('tcp', [4,5,6]), ('targets', [7,8,9])]:
        entity = f'{root}/{group}/{segment("custom/tip")}'
        np.testing.assert_allclose(column_rows(chunks, entity, 'Transform3D:translation')[0][0], expected)
        assert column_rows(chunks, entity, 'Clear:is_recursive') == [[True]]
    geometry_path = f'{root}/visual/{segment("link1:0")}'
    vertices = column_rows(chunks, geometry_path, 'Mesh3D:vertex_positions')
    assert len(vertices) == 1
    normals = column_rows(chunks, geometry_path, 'Mesh3D:vertex_normals')[0]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(vertices[0], [[0,0,0],[2,0,0],[0,3,4]])
    assert column_rows(chunks, geometry_path, 'Transform3D:parent_frame') == [[renderer.frame_id(link)]]
    assert column_rows(chunks, geometry_path, 'CoordinateFrame:frame') == [[renderer.frame_id(geometry_path)]]
    for chunk in chunks:
        if 'Mesh3D:vertex_positions' in chunk.to_record_batch().schema.names:
            assert chunk.is_static
    child1 = column_rows(chunks, link, 'Transform3D:child_frame')[0][0]
    child2 = column_rows(chunks, link2, 'Transform3D:child_frame')[0][0]
    assert child1 != child2
    subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'verify', str(path)], check=True, capture_output=True)
