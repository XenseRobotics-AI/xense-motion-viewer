"""Runnable offline policy producer; native Elite simulation stays on main thread."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..control import JointTarget, RobotSession
from ..policy import JointPreview, PolicyChunk, PolicyInput, SampleTime, StreamEvent
from ..streams import VisualizationWorker


def example_chunk(adapter, *, kind, chunk_id, now, anchor=None, direction=1, side=None):
    """Hand-authored denormalized reference, not inference or a robot planner."""
    state = adapter.snapshot() if anchor is None else anchor
    side = side or next(iter(adapter.capabilities().groups))
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
    from .elite_trajectory import DEFAULT_TRAJECTORY, RecordedTrajectory
    from ..policy_renderer import PolicyRerunRenderer
    parser = argparse.ArgumentParser(description='Offline Elite full policy chunk + native response; no hardware')
    parser.add_argument('--station', choices=['diagonal-07', 'diagonal-08'])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--collision-profile', default='auto')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--kind', choices=['tcp', 'joint'], default='tcp')
    parser.add_argument('--source', choices=['recorded', 'synthetic'],
                        help='Default: recorded for TCP; synthetic for --kind joint (dataset has no joints)')
    parser.add_argument('--trajectory', type=Path, help='Exported episode .npz; default is bundled episode 0')
    parser.add_argument('--arm', choices=['both', 'left', 'right'], default='both',
                        help='Default: synchronized dual-arm playback; left/right selects one arm')
    length = parser.add_mutually_exclusive_group()
    length.add_argument('--steps', type=int, help='Limit native steps; recorded data stops at episode end')
    length.add_argument('--duration', type=float, help='Limit playback seconds; default is the full episode')
    parser.add_argument('--queue-capacity', type=int, default=128)
    viewers = parser.add_mutually_exclusive_group()
    viewers.add_argument('--viewer', dest='viewer', action='store_true')
    viewers.add_argument('--no-viewer', dest='viewer', action='store_false')
    parser.set_defaults(viewer=True)
    parser.add_argument('--fast', action='store_true', help='Finite unpaced verification; default is wall-clock paced')
    parser.add_argument('--port', type=int, default=9877)
    parser.add_argument('--telemetry', action='store_true', help='Optional text panel beside the one shared 3D scene')
    args = parser.parse_args(argv)
    if ((args.steps is not None and args.steps < 2) or args.queue_capacity < 1
            or (args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0))):
        parser.error('--steps must be >=2; --queue-capacity and --duration must be positive')
    source = args.source or ('synthetic' if args.kind == 'joint' else 'recorded')
    if source == 'recorded' and args.kind != 'tcp':
        parser.error('Recorded episode contains TCP actions, not joint angles; use --kind tcp')
    if args.trajectory is not None and source != 'recorded':
        parser.error('--trajectory requires --source recorded')
    arms = ('left', 'right') if args.arm == 'both' else (args.arm,)
    trajectories = ({side: RecordedTrajectory(args.trajectory or DEFAULT_TRAJECTORY, arm=side)
                     for side in arms} if source == 'recorded' else {})
    trajectory = next(iter(trajectories.values()), None)
    if any(t.gripper_openings is None for t in trajectories.values()):
        parser.error('Recorded playback requires <arm>_gripper.pos actions for each selected arm')
    simulation = _simulation(args)
    adapter = EliteStationAdapter(simulation)
    period = next(iter(simulation.arms.values())).config.period
    duration = trajectory.duration if trajectory else 12.
    if args.duration is not None:
        duration = min(duration, args.duration) if trajectory else args.duration
    steps = args.steps if args.steps is not None else max(2, int(np.ceil(duration / period)))
    if trajectory:
        steps = min(steps, int(np.ceil(trajectory.duration / period)))
    session = RobotSession(adapter)
    recording = _recording(args.save, viewer_port=args.port if args.viewer else None)
    renderer = PolicyRerunRenderer(recording, adapter.describe(), compact=True,
                                  shared_view=True, telemetry=args.telemetry, groups=arms)
    reference_source = (trajectory.provenance() if trajectory else
                        dict(source='synthetic', selected_arm=args.arm, description='Hand-authored reference'))
    reference_source.update(selected_arm=args.arm, selected_arms=list(arms))
    if len(arms) > 1:
        reference_source.pop('other_arm', None)
        reference_source['coordination'] = 'Both targets submitted at the same sample time before one native station step'
    recording.log(f'{renderer.panel}/reference_source',
                  renderer.rr.TextDocument(json.dumps(reference_source, indent=2), media_type='text/plain'), static=True)
    worker = VisualizationWorker(lambda event: renderer.consume(event, stats=worker.stats),
                                 capacity=args.queue_capacity, close_callback=renderer.close, coalesce_feedback=True)
    inputs = session.activate('policy')
    policies = {side: PolicyInput(inputs, lambda event, side=side: worker.publish(replace(event, group=side)))
                for side in arms}
    initial = session.snapshot()
    refs = dict.fromkeys(arms)
    submissions = []
    last_indices = dict.fromkeys(arms)
    chunk_index = 0
    next_chunk = 0.
    chunk_start = chunk_end = 0
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
            if trajectory:
                source_index = trajectory.index_at(now)
                if source_index >= chunk_end:
                    chunk_start = source_index
                    for side, policy in policies.items():
                        chunk, chunk_end = trajectories[side].chunk(session.capabilities, chunk_start, now=now)
                        policy.accept(chunk)
                        last_indices[side] = None
            elif now >= next_chunk:
                chunk_index += 1
                for side, policy in policies.items():
                    chunk, preview = example_chunk(adapter, kind=args.kind,
                        chunk_id=f'{args.kind}-{side}-{chunk_index}', now=now, anchor=initial,
                        direction=1 if chunk_index % 2 else -1, side=side)
                    policy.accept(chunk, preview=preview)
                    last_indices[side] = None
                next_chunk = now + 2.5
            # Submit both arms' targets before advancing the shared station.
            # One SDK owner thread and one lease preserve the native dual-arm
            # collision coordinator and use exactly one common sample clock.
            for side, policy in policies.items():
                policy.expire(now)
                chunk = policy.active_chunk
                if chunk is None:
                    continue
                index = (source_index - chunk_start if trajectory else
                         min(int((now - chunk.timing.received_at) / 0.16), len(chunk.values) - 1))
                if index != last_indices[side]:
                    policy.select(index, now=now)
                    submission = policy.execute_selected(now=now)
                    refs[side] = submission.ref
                    submissions.append(dict(arm=side, chunk_id=submission.ref.chunk_id, step_index=submission.ref.step_index,
                                            sequence=submission.receipt.sequence, host_time=now))
                    if trajectory:
                        opening = float(trajectories[side].gripper_openings[source_index])
                        gripper_receipt = inputs.submit(adapter.grippers.target(side, opening))
                        worker.publish(StreamEvent('submitted', SampleTime(now), 'reference', submission.ref,
                            receipt=gripper_receipt, group=side,
                            reason='Gripper action from the same recorded source frame as the TCP target'))
                        submissions[-1].update(source_frame_index=int(trajectory.frame_indices[source_index]),
                                               source_time=float(trajectory.source_times[source_index]),
                                               gripper_opening=opening, gripper_sequence=gripper_receipt.sequence)
                    last_indices[side] = index
            adapter.step()  # native StationSimulation.step; SDK owner thread only
            completed += 1
            state = session.snapshot()
            if step % 10 == 9 or step == steps - 1:
                now = state.time
                # Native planner q is an observable simulation reference, not
                # a transport callback or a successful setter's return value.
                timing = SampleTime(now, state.time, 'elite_simulation')
                for side in arms:
                    names = session.capabilities.groups[side]
                    planned = JointTarget(side, dict(zip(names, simulation.arms[side].planner.q(), strict=True)))
                    worker.publish(StreamEvent('planned', timing, 'simulated', refs[side], command=planned,
                        reason='Native planner.q() after StationSimulation.step; no transport evidence', group=side))
                # Full station feedback has no single-arm step association.
                worker.publish(StreamEvent('feedback', timing, 'simulated', state=state,
                    reason='Snapshot after native simulation step; not hardware measured'))
    except KeyboardInterrupt:
        pass
    finally:
        for policy in policies.values():
            policy.close()
        closed = worker.close(timeout=2)
    final = session.snapshot()
    stats = asdict(worker.stats)
    result = dict(recording=str(args.save.resolve()) if args.save else None, kind=args.kind, native_steps=completed,
        initial_joints=dict(initial.joint_positions), final_joints=dict(final.joint_positions),
        max_joint_response_rad=max(abs(final.joint_positions[n] - initial.joint_positions[n]) for n in final.joint_positions),
        joint_response_by_arm={side: max(abs(final.joint_positions[n] - initial.joint_positions[n])
                                        for n in session.capabilities.groups[side]) for side in arms},
        final_gripper_openings={side: adapter.grippers.opening(side) for side in arms},
        simulation_time=final.time, wall_seconds=time.perf_counter() - wall_start, paced=not args.fast,
        shared_3d_view=True, submissions=submissions, display=stats,
        display_closed=closed, sent='unavailable: no transport send telemetry',
        reference_source=reference_source,
        origin=f'offline simulated response / {source} reference', hardware_connections=0)
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
