import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip('libpyelite')
from libpyelite.sim import StationSimulation
from robot_motion_workbench.adapters.elite import EliteStationAdapter
from robot_motion_workbench.urdf import URDFModel

pytestmark = pytest.mark.elite


@pytest.mark.parametrize('station', ['diagonal-07', 'diagonal-08'])
def test_production_pinocchio_link_tool_and_collision_parity(station):
    import pinocchio as pin
    sim = StationSimulation(station, collision_profile='auto')
    adapter = EliteStationAdapter(sim)
    description = adapter.describe()
    model = URDFModel(description)
    # Adapter construction/snapshot cannot reset or step the simulation.
    before = sim.q.copy()
    state = adapter.snapshot()
    np.testing.assert_array_equal(sim.q, before)
    assert state.time == 0
    assert description.dof == len(sim.q) + 2  # Independent gripper drivers; mimic fingers follow in the URDF.
    data = sim.model.createData()
    for delta in [0, 0.03]:
        q = sim.viewer_q().copy()
        for indices in sim.indices.values():
            q[indices] += delta
        named = dict(state.joint_positions)
        for side, indices in sim.indices.items():
            for joint, index in zip(sim.joint_names, indices, strict=True):
                named[f'{side}_{joint}'] = q[index]
        actual = model.link_transforms(named)
        pin.forwardKinematics(sim.model, data, q)
        pin.updateFramePlacements(sim.model, data)
        for link in model.links:
            np.testing.assert_allclose(actual[link], data.oMf[sim.model.getFrameId(link)].homogeneous, atol=2e-12)
        native_placements = sim.collision.placements(q)
        assert len(native_placements) == len(description.collision_meshes)
        for mesh, placement in zip(description.collision_meshes, native_placements, strict=True):
            np.testing.assert_allclose(actual[mesh.parent_link] @ mesh.link_from_mesh, placement.homogeneous, atol=2e-12)
        if delta == 0:
            for side, tool in description.tool_frames.items():
                np.testing.assert_allclose(actual[tool.parent_link] @ tool.link_from_tool, state.tcp_poses[side], atol=2e-12)
    for shape, mesh in zip(sim.collision.shapes, description.collision_meshes, strict=True):
        if shape['type'] == 'capsule':
            np.testing.assert_allclose([mesh.vertices[:,2].min(), mesh.vertices[:,2].max()],
                [-shape['length']/2-shape['radius'], shape['length']/2+shape['radius']], atol=1e-8)


def test_qt_entry_offscreen(tmp_path):
    path = tmp_path/'qt.rrd'
    env = os.environ | {'QT_QPA_PLATFORM': 'offscreen'}
    process = subprocess.run([sys.executable, '-m', 'robot_motion_workbench.adapters.elite',
        '--no-viewer', '--no-show', '--display-hz', '60', '--save', str(path), '--collision-profile', 'auto'],
        env=env, text=True, capture_output=True, timeout=120)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.strip().splitlines()[-1])
    assert result['qt_smoke'] == 'passed'
    assert result['simulation_seconds'] > 0
    assert all(result[key] for key in ['joint_input_checked','cartesian_input_checked','gripper_input_checked',
                                      'ik_gripper_input_checked',
                                      'collision_toggle_checked','reset_checked'])
    assert path.stat().st_size > 1000
    subprocess.run([sys.executable, '-m', 'rerun', 'rrd', 'verify', str(path)], check=True, capture_output=True)
