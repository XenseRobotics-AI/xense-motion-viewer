import numpy as np
import pytest

from robot_motion_workbench import RobotDescription, RobotState, ToolFrame
from robot_motion_workbench.urdf import URDFModel


def test_snapshots_are_defensive_and_rigid(robot_urdf, names):
    source = np.eye(4)
    joints = {n: 0 for n in names}
    state = RobotState(1, joints, source, {'gripper': source})
    source[0, 3] = 4
    joints[names[0]] = 3
    assert state.base_transform[0, 3] == state.tcp_poses['gripper'][0, 3] == state.joint_positions[names[0]] == 0
    with pytest.raises(ValueError):
        state.base_transform.setflags(write=True)
    with pytest.raises(TypeError):
        state.joint_positions[names[0]] = 4
    with pytest.raises(ValueError):
        RobotState(0, {}, np.diag([2, 1, 1, 1]))
    with pytest.raises(ValueError):
        RobotState(float('nan'), {})
    with pytest.raises(ValueError):
        RobotState(0, {'a': float('inf')})
    with pytest.raises(ValueError):
        RobotDescription('robot', robot_urdf, ('a', 'a'))


def test_seven_dof_axis_origin_mimic_and_tool(robot_urdf, names):
    base = np.eye(4)
    base[0, 3] = 5
    desc = RobotDescription('seven', robot_urdf, names, base, tool_frames={'tip': ToolFrame('tool')})
    model = URDFModel(desc)
    q = dict.fromkeys(names, 0)
    q[names[0]], q[names[1]] = np.pi/2, 2
    fk = model.link_transforms(q)
    # Rz(pi/2) Rx(pi/2) and prismatic along local +Y => world +Z.
    expected_r = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(fk['link1'][:3, :3], expected_r, atol=1e-15)
    np.testing.assert_allclose(fk['link1'][:3, 3], [6, 0, 2], atol=1e-15)
    assert desc.dof == 7
    assert model.resolve_positions(q)['follower'] == pytest.approx(-np.pi/4 + .1)
    angle = -np.pi/4 + .1
    rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    np.testing.assert_allclose(fk['tool'][:3, :3], expected_r @ rz, atol=1e-15)
    np.testing.assert_allclose(fk['tool'][:3, 3], [6.2, 0, 2], atol=1e-15)


def test_mesh_scale_visual_offset_and_material(robot_urdf, names):
    model = URDFModel(RobotDescription('robot', robot_urdf, names))
    geometry = model.load_geometry()
    mesh = geometry[0][1]
    np.testing.assert_allclose(mesh.vertices, [[0, 0, 0], [2, 0, 0], [0, 3, 4]])
    np.testing.assert_allclose(mesh.link_from_mesh[:3, 3], [.2, .3, .4])
    np.testing.assert_allclose(mesh.link_from_mesh[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    assert mesh.rgba == (1, 0, 0, 1)
    assert model.source_mesh_count == 1


def test_scene_nodes_are_baked_once(tmp_path):
    import trimesh
    scene = trimesh.Scene()
    t = np.eye(4); t[0, 3] = 3
    scene.add_geometry(trimesh.creation.box(extents=[1, 1, 1]), transform=t)
    scene.export(tmp_path/'scene.glb')
    urdf = tmp_path/'model.urdf'
    urdf.write_text('<robot name="fixture"><link name="base"><visual><geometry><mesh filename="scene.glb" scale="2 1 1"/></geometry></visual></link></robot>')
    model = URDFModel(RobotDescription('fixture', urdf, ()))
    mesh = model.load_geometry()[0][1]
    np.testing.assert_allclose([mesh.vertices[:,0].min(), mesh.vertices[:,0].max()], [5, 7])


@pytest.mark.parametrize('mutation', ['floating', 'mimic_cycle', 'graph_cycle', 'missing_joint', 'zero_axis'])
def test_rejects_ambiguous_models(robot_urdf, names, mutation):
    text = robot_urdf.read_text()
    if mutation == 'floating': text = text.replace('type="prismatic"', 'type="floating"')
    if mutation == 'mimic_cycle': text = text.replace('mimic joint="axis/0"', 'mimic joint="follower"')
    if mutation == 'graph_cycle': text = text.replace('<parent link="root"/>', '<parent link="link6"/>')
    if mutation == 'zero_axis': text = text.replace('xyz="2 0 0"', 'xyz="0 0 0"')
    robot_urdf.write_text(text)
    with pytest.raises(ValueError):
        URDFModel(RobotDescription('bad', robot_urdf, names[:-1] if mutation == 'missing_joint' else names))


def test_package_uri_resolution(robot_urdf, names):
    robot_urdf.write_text(robot_urdf.read_text().replace('filename="triangle.obj"', 'filename="package://meshes/triangle.obj"'))
    desc = RobotDescription('pkg', robot_urdf, names, package_roots={'meshes': robot_urdf.parent})
    assert len(URDFModel(desc).load_geometry()) == 2
    with pytest.raises(ValueError, match='package_roots'):
        URDFModel(RobotDescription('pkg', robot_urdf, names)).load_geometry()
