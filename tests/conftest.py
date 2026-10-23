from pathlib import Path
import pytest


@pytest.fixture
def robot_urdf(tmp_path):
    # Seven independently named joints plus a mimic and a fixed tool. A mesh
    # has a nontrivial local offset, unit scale and primitive companion.
    (tmp_path / 'triangle.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 1\nf 1 2 3\n')
    links = ['<link name="root"/>']
    for i in range(7):
        visual = '''<visual><origin xyz="0.2 0.3 0.4" rpy="0 0 1.5707963267948966"/>
        <geometry><mesh filename="triangle.obj" scale="2 3 4"/></geometry>
        <material name="red"/></visual>''' if i == 1 else ''
        links.append(f'<link name="link{i}">{visual}</link>')
    links += ['<link name="mimic_link"/>', '<link name="tool"><visual><geometry><box size="0.1 0.2 0.3"/></geometry></visual></link>']
    joints = []
    for i in range(7):
        parent = 'root' if i == 0 else f'link{i-1}'
        origin = '<origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>' if i == 0 else ''
        joints.append(f'''<joint name="axis/{i}" type="{'prismatic' if i == 1 else 'revolute'}">
        <parent link="{parent}"/><child link="link{i}"/>{origin}
        <axis xyz="{'0 2 0' if i == 1 else '2 0 0'}"/><limit lower="-10" upper="10" effort="10" velocity="10"/></joint>''')
    joints.append('''<joint name="follower" type="revolute"><parent link="link6"/><child link="mimic_link"/>
    <axis xyz="0 0 1"/><mimic joint="axis/0" multiplier="-0.5" offset="0.1"/></joint>''')
    joints.append('''<joint name="tool_fixed" type="fixed"><parent link="mimic_link"/><child link="tool"/>
    <origin xyz="0 0 0.2"/></joint>''')
    path = tmp_path/'model.urdf'
    path.write_text('<robot name="generic"><material name="red"><color rgba="1 0 0 1"/></material>'+''.join(links+joints)+'</robot>')
    return path


@pytest.fixture
def names():
    return tuple(f'axis/{i}' for i in range(7))
