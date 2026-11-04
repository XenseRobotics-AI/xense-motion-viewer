"""Model selection and real DH opening motion through the viewer adapter."""
import numpy as np
import pytest

pytest.importorskip('libpyelite')
from libpyelite.sim import StationSimulation
from xense_motion_viewer.adapters.elite import EliteStationAdapter, _parser, _simulation


def test_cli_selects_dh_without_changing_taccap_default():
    parser = _parser(qt=True)
    dh = _simulation(parser.parse_args(['--station', 'diagonal-08', '--end-effector', 'dh-ag95', '--collision-profile', 'auto']))
    assert dh.assembly.name == 'diagonal-08-dh-ag95'
    assert dh.collision is not None and dh.collision_error is None
    assert _simulation(parser.parse_args([])).assembly.name == 'diagonal-08-taccap'


@pytest.mark.parametrize('model', ['diagonal-08-taccap', 'diagonal-08-dh-ag95'])
def test_open_close_direction_and_mimic_poses_reach_the_native_guard(model):
    sim = StationSimulation(model, collision_profile='auto')
    adapter = EliteStationAdapter(sim)
    fingers = adapter.grippers
    for opening in (.5, 0., 1.):
        for side in sim.arms:
            adapter.submit(fingers.target(side, opening))
        for _ in range(200):
            adapter.step()
        for side in sim.arms:
            assert not fingers.blocked
            assert fingers.opening(side) == pytest.approx(opening, abs=1e-10)
            q = fingers.positions[fingers.drivers[side]]
            if sim.assembly.end_effector == 'dh-ag95':
                assert q == pytest.approx((1 - opening) * .6524)
        assert sim.collision.check(fingers.full_q()).safe
    arm_state = sim.q.copy()
    adapter.reset()
    np.testing.assert_allclose(sim.q, sim.home_q)
    np.testing.assert_allclose(sim.q, arm_state)
    assert all(fingers.opening(side) == 1 for side in sim.arms)
