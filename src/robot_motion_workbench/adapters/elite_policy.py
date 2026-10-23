"""Runnable offline policy producer; native Elite simulation stays on main thread."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..control import JointTarget, RobotSession
from ..policy import JointPreview, PolicyChunk, PolicyInput, SampleTime, StreamEvent
from ..streams import VisualizationWorker


def example_chunk(adapter, *, kind, chunk_id, now, anchor=None, direction=1):
    """Hand-authored denormalized reference, not inference or a robot planner."""
    state = adapter.snapshot() if anchor is None else anchor
    side = next(iter(adapter.capabilities().groups))
    names = adapter.capabilities().groups[side]
    offsets = np.linspace(0.2, 2.4, 12)
    fraction = np.linspace(0, 1, len(offsets))
    preview = None
    if kind == 'joint':
        values = np.tile([state.joint_positions[n] for n in names], (len(offsets), 1))
        values[:, 0] += direction * 0.06 * fraction
        values[:, 2] += 0.025 * np.sin(fraction * np.pi)
        semantics, frame, tcp = 'absolute_joint_positions', 'joint', None
        full_names = adapter.describe().joint_names
        full = np.tile([state.joint_positions[n] for n in full_names], (len(offsets), 1))
        for index, joint in enumerate(names):
            full[:, full_names.index(joint)] = values[:, index]
        # Elite owns the group mapping. This preview explicitly holds the other
        # arm at the current snapshot; no future measurement or Cartesian IK.
        preview = JointPreview(full_names, full, 'Elite example: joint references; other arm held at snapshot')
    else:
        values = np.tile(state.tcp_poses[side], (len(offsets), 1, 1))
        values[:, 0, 3] += direction * 0.06 * fraction
        values[:, 1, 3] += 0.03 * np.sin(fraction * np.pi)
        values[:, 2, 3] += 0.04 * np.sin(fraction * np.pi)
        semantics, frame, tcp = 'absolute_tcp_poses', 'world', side
    return PolicyChunk(chunk_id, side, names, offsets, values,
        SampleTime(now), now + 3.0, semantics, frame, tcp), preview


def main(argv=None):
    from .elite import EliteStationAdapter, _recording, _simulation
    from ..policy_renderer import PolicyRerunRenderer
    parser = argparse.ArgumentParser(description='Offline Elite full policy chunk + native response; no hardware')
    parser.add_argument('--station', choices=['diagonal-07', 'diagonal-08'])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--collision-profile', default='auto')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--kind', choices=['tcp', 'joint'], default='tcp')
    parser.add_argument('--steps', type=int, help='Finite native steps; default is 12 seconds of simulation')
    parser.add_argument('--queue-capacity', type=int, default=128)
    viewers = parser.add_mutually_exclusive_group()
    viewers.add_argument('--viewer', dest='viewer', action='store_true')
    viewers.add_argument('--no-viewer', dest='viewer', action='store_false')
    parser.set_defaults(viewer=True)
    parser.add_argument('--fast', action='store_true', help='Finite unpaced verification; default is wall-clock paced')
    parser.add_argument('--port', type=int, default=9877)
    parser.add_argument('--telemetry', action='store_true', help='Optional text panel beside the one shared 3D scene')
    args = parser.parse_args(argv)
    if (args.steps is not None and args.steps < 2) or args.queue_capacity < 1:
        parser.error('--steps must be >=2 and --queue-capacity must be positive')
    simulation = _simulation(args)
    adapter = EliteStationAdapter(simulation)
    period = next(iter(simulation.arms.values())).config.period
    steps = args.steps if args.steps is not None else round(12. / period)
    session = RobotSession(adapter)
    recording = _recording(args.save, viewer_port=args.port if args.viewer else None)
    renderer = PolicyRerunRenderer(recording, adapter.describe(), compact=True,
                                  shared_view=True, telemetry=args.telemetry)
    worker = VisualizationWorker(lambda event: renderer.consume(event, stats=worker.stats),
                                 capacity=args.queue_capacity, close_callback=renderer.close, coalesce_feedback=True)
    policy = PolicyInput(session.activate('policy'), worker.publish)
    initial = session.snapshot()
    ref = None
    submissions = []
    last_index = None
    chunk_index = 0
    next_chunk = 0.
    completed = 0
    wall_start = time.perf_counter()
    next_tick = wall_start
    try:
        worker.publish(StreamEvent('feedback', SampleTime(0, initial.time, 'elite_simulation'), 'simulated', state=initial))
        for step in range(steps):
            if not args.fast:
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -.1:
                    next_tick = time.perf_counter()  # Do not race through old wall-clock debt.
                next_tick += period
            state = session.snapshot()
            now = state.time  # explicit offline host clock, advanced by native sim
            if now >= next_chunk:
                chunk_index += 1
                chunk, preview = example_chunk(adapter, kind=args.kind,
                    chunk_id=f'{args.kind}-{chunk_index}', now=now, anchor=initial,
                    direction=1 if chunk_index % 2 else -1)
                policy.accept(chunk, preview=preview)
                next_chunk = now + 2.5
                last_index = None
            policy.expire(now)
            chunk = policy.active_chunk
            if chunk is not None:
                # This loop is example-specific host policy, not a generic
                # controller. The native planner still generates every sample.
                index = min(int((now - chunk.timing.received_at) / 0.16), len(chunk.values) - 1)
                if index != last_index:
                    policy.select(index, now=now)
                    submission = policy.execute_selected(now=now)
                    ref = submission.ref
                    submissions.append(dict(chunk_id=ref.chunk_id, step_index=ref.step_index,
                                            sequence=submission.receipt.sequence, host_time=now))
                    last_index = index
            adapter.step()  # native StationSimulation.step; SDK owner thread only
            completed += 1
            state = session.snapshot()
            if step % 10 == 9 or step == steps - 1:
                now = state.time
                side = next(iter(simulation.arms))
                names = session.capabilities.groups[side]
                # Native planner q is an observable simulation reference, not
                # a transport callback or a successful setter's return value.
                planned = JointTarget(side, dict(zip(names, simulation.arms[side].planner.q(), strict=True)))
                timing = SampleTime(now, state.time, 'elite_simulation')
                worker.publish(StreamEvent('planned', timing, 'simulated', ref, command=planned,
                    reason='Native planner.q() after StationSimulation.step; no transport evidence'))
                worker.publish(StreamEvent('feedback', timing, 'simulated', ref, state=state,
                    reason='Snapshot after native simulation step; not hardware measured'))
    except KeyboardInterrupt:
        pass
    finally:
        policy.close()
        closed = worker.close(timeout=2)
    final = session.snapshot()
    stats = asdict(worker.stats)
    result = dict(recording=str(args.save.resolve()) if args.save else None, kind=args.kind, native_steps=completed,
        initial_joints=dict(initial.joint_positions), final_joints=dict(final.joint_positions),
        max_joint_response_rad=max(abs(final.joint_positions[n] - initial.joint_positions[n]) for n in final.joint_positions),
        simulation_time=final.time, wall_seconds=time.perf_counter() - wall_start, paced=not args.fast,
        shared_3d_view=True, submissions=submissions, display=stats,
        display_closed=closed, sent='unavailable: no transport send telemetry',
        origin='offline simulated response / hand-authored reference', hardware_connections=0)
    if args.save:
        args.save.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not closed or stats['errors']:
        raise RuntimeError(f'Display failed: {stats}')
    return 0


def replay_main(argv=None):
    """Open a recording in Rerun. No simulation or control object is created."""
    parser = argparse.ArgumentParser(description='Read-only Rerun policy recording replay')
    parser.add_argument('recording', type=Path)
    args = parser.parse_args(argv)
    return subprocess.run([sys.executable, '-m', 'rerun', str(args.recording.resolve(strict=True))], check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
