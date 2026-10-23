import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from robot_motion_workbench.adapters.reference import demo_urdf


@pytest.mark.parametrize('model', ['portable', 'rizon4'])
def test_generic_qt_all_axes_policy_takeover(tmp_path, model):
    if importlib.util.find_spec('PySide6') is None:
        pytest.skip('Optional Qt dependency is not installed')
    path = demo_urdf()
    if model == 'rizon4':
        if not os.environ.get('RMW_RIZON4_URDF'):
            pytest.skip('Set RMW_RIZON4_URDF for external asset acceptance')
        path = Path(os.environ['RMW_RIZON4_URDF'])
    qt = Path(importlib.util.find_spec('PySide6').origin).parent / 'Qt'
    env = os.environ | {'QT_QPA_PLATFORM': 'offscreen', 'QT_PLUGIN_PATH': str(qt/'plugins'),
        'QT_QPA_PLATFORM_PLUGIN_PATH': str(qt/'plugins/platforms'),
        'LD_LIBRARY_PATH': str(qt/'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('external_qt_probe.py')),
        str(path), str(tmp_path/'qt-result.json')], env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
