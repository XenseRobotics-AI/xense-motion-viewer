from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('libpyelite')
from libpyelite.sim import StationSimulation
from robot_motion_workbench import RobotSession
from robot_motion_workbench.adapters.elite import EliteStationAdapter
from robot_motion_workbench.urdf import URDFModel

pytestmark = pytest.mark.elite


@pytest.mark.parametrize('station', ['diagonal-07', 'diagonal-08'])
@pytest.mark.parametrize('collision', [None, 'auto'])
def test_fingers_follow_opening_commands_and_share_native_collision_geometry(station, collision):
    import pinocchio as pin
    sim = StationSimulation(station, collision_profile=collision)
    adapter = EliteStationAdapter(sim)
    assert EliteStationAdapter(sim).grippers is adapter.grippers
    inputs = RobotSession(adapter).activate('test')
    fingers = adapter.grippers
    original_tcps = sim.tcp_poses()
    initial = fingers.positions.copy()
    inputs.submit(fingers.target('left', 0))
    inputs.submit(fingers.target('right', .5))
    assert fingers.positions == initial  # Submission alone never teleports the jaws.
    period = next(iter(sim.arms.values())).config.period
    for _ in range(130):
        before = fingers.positions.copy()
        adapter.step()
        assert all(abs(fingers.positions[n] - q) <= fingers.velocities[n] * period + 1e-12
                   for n, q in before.items())
    assert fingers.opening('left') == pytest.approx(0)
    assert fingers.opening('right') == pytest.approx(.5)
    for side in sim.arms:
        np.testing.assert_allclose(sim.tcp_poses()[side], original_tcps[side], atol=1e-10)
    if sim.guard:
        # Use the guard's actual configuration, not another copy of our mapping.
        fingers._publish()
        q = np.asarray(sim.guard.station_q([arm.q for arm in sim.arms.values()]))
        np.testing.assert_allclose(q, fingers.full_q(), atol=1e-12)
        assert sim.collision.check(q).safe
    else:
        q = fingers.full_q()
    state = adapter.snapshot()
    model = URDFModel(adapter.describe())
    placements = model.link_transforms(state.joint_positions)
    native = sim.model.createData()
    pin.forwardKinematics(sim.model, native, q)
    pin.updateFramePlacements(sim.model, native)
    for name in fingers.expanded():
        child = model.joints[name].child
        np.testing.assert_allclose(placements[child], native.oMf[sim.model.getFrameId(child)].homogeneous, atol=2e-12)
    adapter.reset()
    adapter.step()
    assert fingers.positions == initial and fingers.targets == initial


def test_collision_rejected_finger_segment_holds_without_moving_other_joints(monkeypatch):
    sim = StationSimulation(collision_profile='auto')
    adapter = EliteStationAdapter(sim)
    inputs = RobotSession(adapter).activate('test')
    inputs.submit(adapter.grippers.target('left', 0))
    initial = adapter.grippers.positions.copy()
    checked = []
    def blocked(start, end):
        checked.append((start.copy(), end.copy()))
        return SimpleNamespace(safe=False)
    monkeypatch.setattr(sim.collision, 'check_segment', blocked)
    adapter.step()
    assert len(checked) == 1 and not np.array_equal(*checked[0])
    assert adapter.grippers.blocked and adapter.grippers.positions == initial
    assert sim.last_collision.safe


@pytest.mark.parametrize('value', [-.01, 1.01, float('nan')])
def test_invalid_gripper_opening_does_not_change_target(value):
    adapter = EliteStationAdapter(StationSimulation())
    before = adapter.grippers.targets.copy()
    with pytest.raises(ValueError, match='opening'):
        adapter.grippers.target('left', value)
    assert adapter.grippers.targets == before
