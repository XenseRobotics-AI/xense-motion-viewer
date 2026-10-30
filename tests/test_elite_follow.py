import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.elite
@pytest.mark.parametrize('change', ['home', 'calibration', 'velocity', 'world_frame'])
def test_environment_identity_includes_gripper_geometry_and_motion(change):
    pytest.importorskip('libpyelite')
    pytest.importorskip('PySide6')
    from libpyelite.sim import StationSimulation
    from robot_motion_workbench.adapters.elite import EliteStationAdapter
    from robot_motion_workbench.adapters.elite_follow import build_info
    sim = StationSimulation(collision_profile='auto')
    adapter = EliteStationAdapter(sim)
    original = build_info(sim, adapter.grippers)
    joint = adapter.grippers.drivers['left']
    if change == 'home':
        adapter.grippers.home[joint] = 0.
    elif change == 'calibration':
        adapter.grippers.knots['left'][-1, 1] *= .95
    elif change == 'velocity':
        adapter.grippers.velocities[joint] *= .5
    else:
        original_transform = sim.assembly.world_from_base
        def shifted(side):
            value = original_transform(side).copy()
            value[0, 3] += .1
            return value
        sim.assembly.world_from_base = shifted
    changed = build_info(sim, adapter.grippers)
    assert changed['environment_sha256'] != original['environment_sha256']
    assert changed['native_sha256'] == original['native_sha256']


@pytest.mark.elite
def test_follow_qt_and_native_replay(tmp_path):
    pytest.importorskip('libpyelite')
    spec = importlib.util.find_spec('PySide6')
    if spec is None:
        pytest.skip('Optional Qt dependency is not installed')
    root = Path(spec.origin).parent / 'Qt'
    env = os.environ | {'QT_QPA_PLATFORM': 'offscreen',
        'QT_PLUGIN_PATH': str(root / 'plugins'),
        'QT_QPA_PLATFORM_PLUGIN_PATH': str(root / 'plugins/platforms'),
        'LD_LIBRARY_PATH': str(root / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('elite_follow_probe.py')), str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
