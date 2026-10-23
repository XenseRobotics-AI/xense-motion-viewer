import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('probe', ['qt_control_probe.py', 'elite_qt_control_probe.py'])
def test_offscreen_inputs(tmp_path, probe):
    if probe.startswith('elite') and importlib.util.find_spec('libpyelite') is None:
        pytest.skip('Optional Elite dependency is not installed')
    spec = importlib.util.find_spec('PySide6')
    if spec is None:
        pytest.skip('Optional Qt dependency is not installed')
    root = Path(spec.origin).parent / 'Qt'
    env = os.environ | {'QT_QPA_PLATFORM': 'offscreen',
        'QT_PLUGIN_PATH': str(root / 'plugins'),
        'QT_QPA_PLATFORM_PLUGIN_PATH': str(root / 'plugins/platforms'),
        'LD_LIBRARY_PATH': str(root / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name(probe)), str(tmp_path)],
                            env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
