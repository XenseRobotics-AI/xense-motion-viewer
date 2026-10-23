"""Run with python -I -S; only the supplied private install target is added."""
import importlib.util
import json
from pathlib import Path
import site
import sys

target = Path(sys.argv[1]).resolve()
site.addsitedir(str(target))
optional = ('PySide6', 'rerun', 'libpyelite', 'pinocchio', 'trimesh', 'yaml')
assert all(importlib.util.find_spec(name) is None for name in optional)
import robot_motion_workbench as rmw
assert Path(rmw.__file__).is_relative_to(target)
import numpy as np
names = tuple(f'axis/{i}' for i in range(7))
urdf = target.parent / 'isolated-test.urdf'
urdf.write_text('<robot name="isolated"><link name="root"/></robot>')
description = rmw.RobotDescription('isolated', urdf, names)
metadata = rmw.ControlCapabilities({n: rmw.JointControl() for n in names}, {'arm': names})
accepted = []
adapter = rmw.CallbackRobotAdapter(description, metadata,
    read_state=lambda: rmw.RobotState(0, dict.fromkeys(names, 0)), submit_target=accepted.append)
session = rmw.RobotSession(adapter)
receipt = session.activate('policy').submit(rmw.JointTarget('arm', dict.fromkeys(reversed(names), .01)))
assert receipt.status == 'submitted' and len(accepted) == 1
assert tuple(accepted[0].positions) == names
assert session.snapshot().joint_positions[names[0]] == 0
# Lazy import and a useful diagnostic when the optional Elite feature is selected.
from robot_motion_workbench.adapters.elite import _simulation
try:
    _simulation(None)
except RuntimeError as exc:
    diagnostic = str(exc)
    assert 'current-production-source' in diagnostic and 'README.md' in diagnostic
    assert isinstance(exc.__cause__, ModuleNotFoundError)
else:
    raise AssertionError('Elite should not be available')
assert not set(optional).intersection(sys.modules)
print(json.dumps(dict(module=rmw.__file__, sys_path=sys.path,
    absent_optional_dependencies=optional, axes=len(names), receipt=receipt.status,
    observed_unchanged=True, elite_diagnostic=diagnostic), indent=2))
