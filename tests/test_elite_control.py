from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('libpyelite')
from libpyelite.sim import StationSimulation
from libpyelite.control_profile import describe_config
from xense_motion_viewer import JointTarget, RobotSession, TcpTarget
from xense_motion_viewer.adapters.elite import EliteStationAdapter, _simulation

pytestmark = pytest.mark.elite


@pytest.mark.parametrize('station', ['diagonal-07', 'diagonal-08'])
@pytest.mark.parametrize('collision', [None, 'auto'])
def test_public_targets_retain_current_native_planner_and_settings(station, collision, tmp_path):
    reference = StationSimulation(station)
    settings = deepcopy(reference.settings)
    settings.update(native_ik_max_iterations=17,
                    native_max_target_joint_jump=.15, native_max_tcp_linear_velocity=.07)
    tool = np.eye(4); tool[2, 3] = .125
    settings['left_tool_transform'] = tool.tolist()
    settings['right_tool_transform'] = tool.tolist()
    settings['native_assembly'] = station
    settings['type'] = 'bi_elite_cs66_rt'
    settings['native_collision_profile'] = collision
    recipe = tmp_path / 'recipe.json'
    # Existing load_recipe supports a YAML document; JSON is a YAML subset.
    recipe.write_text(json.dumps({'robot': settings}))
    args = SimpleNamespace(recipe=recipe, station=station, collision_profile=None)
    actual = _simulation(args)
    direct = StationSimulation(station, settings=settings, collision_profile=collision)
    adapter = EliteStationAdapter(actual)
    session = RobotSession(adapter)
    producer = session.activate('policy')
    for side in actual.arms:
        assert describe_config(actual.configs[side]) == describe_config(direct.configs[side])
        np.testing.assert_array_equal(actual.arms[side].tool, tool)
    before = actual.q.copy()
    for side in actual.arms:
        names = session.capabilities.groups[side]
        q = actual.arms[side].q.copy()
        q[0] += .015
        producer.submit(JointTarget(side, dict(reversed(list(zip(names, q, strict=True))))))
        direct.set_joint_target(side, q)
    np.testing.assert_array_equal(actual.q, before)
    max_motion = 0.
    for step in range(80):
        if step in (20, 45):
            for side in actual.arms:
                pose = direct.tcp_poses()[side].copy()
                pose[0, 3] += .003 if step == 20 else -.002
                producer.submit(TcpTarget(side, pose))
                direct.set_target(side, pose)
        a, b = adapter.step(), direct.step()
        np.testing.assert_array_equal(actual.q, direct.q)
        max_motion = max(max_motion, float(np.max(np.abs(actual.q - before))))
        for side in actual.arms:
            np.testing.assert_array_equal(actual.targets()[side], direct.targets()[side])
            # Wall-clock solve duration is intentionally nondeterministic.
            assert {k: v for k, v in a[side].items() if k != 'solve_time_s'} == {
                k: v for k, v in b[side].items() if k != 'solve_time_s'}
            if collision:
                assert 'collision_constraints' in a[side]
                assert actual.last_collision.safe
    assert max_motion > 1e-5
    assert session.last_receipt.sequence == 6
    # Native jump policy still rejects otherwise statically valid targets.
    side = next(iter(actual.arms))
    names = session.capabilities.groups[side]
    q = actual.arms[side].q.copy(); q[0] += .2
    with pytest.raises(ValueError, match='jump'):
        producer.submit(JointTarget(side, dict(zip(names, q, strict=True))))
    assert session.last_receipt.sequence == 6


def test_collision_failure_latches_and_adapter_does_not_bypass_it():
    sim = StationSimulation(collision_profile='auto')
    adapter = EliteStationAdapter(sim)
    producer = RobotSession(adapter).activate('programmatic')
    sim.collision_error = 'test collision latch'
    producer.submit(TcpTarget('left', sim.tcp_poses()['left']))
    with pytest.raises(RuntimeError, match='reset explicitly'):
        adapter.step()
    assert sim.collision_error == 'test collision latch'
    sim.reset_to_home()
    adapter.step()


def test_recipe_collision_override_and_station_mismatch(tmp_path):
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text('robot:\n  type: bi_elite_cs66_rt\n  native_assembly: diagonal-07\n  native_collision_profile: auto\n')
    args = SimpleNamespace(recipe=recipe, station=None, collision_profile='off')
    assert _simulation(args).collision is None
    args.station = 'diagonal-08'
    with pytest.raises(ValueError, match='must agree'):
        _simulation(args)
