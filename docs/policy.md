# Policy telemetry and lifecycle

`PolicyInput` uses an already-acquired Phase 2 `InputSession`. Accepting chunks,
selecting steps, displaying, and replaying are read-only operations with respect
to the robot. Only `execute_selected(now=...)` calls that session's `submit`.
Offsets describe intent and do not schedule a control loop. The Elite example's
host loop calls unchanged native simulation; the portable/external-URDF example
explicitly advances a synthetic numerical follower with no dynamics or IK.

`SampleTime.received_at` is the host elapsed receipt/availability clock. Optional
`source_time` and `source_clock` must appear together; clocks are never silently
subtracted. The Elite demo explicitly uses native simulation elapsed time as its
offline host clock, while retaining the named `elite_simulation` source clock.
External hosts must establish their own clock synchronization. Recording epochs
must be monotonic; use a new recording for a new epoch.

## Separate observations

Prediction, selection, submitted request, planned, sent and feedback have distinct
paths and immutable values. `PolicySubmission.ref` includes chunk ID and selected
step index. This association comes from the SDK/host; it cannot be deduced from
timestamps or nearest predictions. Feedback with unknown association uses
`ref=None`. Lifecycle events with no selected step use `StepRef(chunk_id, None)`.

The host constructs events from copies on the SDK's owner thread:

```python
from robot_motion_workbench import SampleTime, StreamEvent

snapshot = session.snapshot()
display_worker.publish(StreamEvent(
    'feedback', SampleTime(host_receipt),  # Source sample time unavailable.
    'simulated', submitted.ref, state=snapshot,
))
# An SDK-owned planner callback may provide its actual named planned target:
display_worker.publish(StreamEvent(
    'planned', SampleTime(host_receipt), 'reference', submitted.ref,
    command=actual_planned_target,
))
# Only a successful send callback with evidence of the actual sample may emit:
display_worker.publish(StreamEvent(
    'sent', SampleTime(send_receipt, source_time=send_source_time,
                       source_clock='sdk_send_clock'),
    'measured', submitted.ref, command=actual_sent_target,
))
```

Only provide `SampleTime(host_receipt, source_time=actual_source_time,
source_clock='sdk_elapsed')` when that source sample time is exposed by the SDK.
In particular, the injected Flexiv joint bridge exposes no such timestamp:
`snapshot.time` is elapsed host poll completion, so it must not be reused as
a source timestamp. A recent poll alone does not establish sample freshness.
The serialized feedback `state_time` is the adapter's rendering clock; only
`timing.source_time` / `timing.source_clock` declare source-sample timing.
Older 0.3 recordings called this neutral state field `source_state_time`;
0.4 removes that misleading label without altering rendering or stream clocks.

This is an integration recipe, not an implemented hardware transport. Use
`origin='simulated'` for simulation send/feedback samples, `measured` for actual
observed samples, and `reference` for policy/planned/request data. Policy chunks
are always references. Future references cannot become measured `RobotState`
samples. Available `sent` and `feedback` events reject `reference` provenance.

A setter's success creates only the session receipt and an unavailable sent
fallback with `fallback=True` and that receipt's `submission_sequence`. This is
submission bookkeeping, not an authoritative report that a transport sent
nothing. It never fabricates a sent/planned/feedback value. Without telemetry:

```python
display_worker.publish(StreamEvent(
    'planned', SampleTime(host_receipt), 'reference', submitted.ref,
    reason='Adapter has no planner telemetry',
))
```

Unavailable streams have no command/state value and require an explanation.
The renderer labels absent/unavailable streams, host receipt times, known or
unknown source time, origin, association and freshness independently. RRD data
retains every supplied joint/pose value even when the compact status panel shows
only a representative value. Blue TCP paths show the remaining selected chunk;
green lines and robot geometry are received feedback. The orange target marker
has no text label, and the complete chunk remains in the recording metadata.
A TCP close-up shows short paths.

Unavailable feedback updates the availability/reason table but does not relabel
retained geometry. The renderer separately records `rendered_feedback` with the
origin, host/source times and age of the last available state. A transition
between available simulated and measured states clears TCP history before the
new point, so a displayed line never joins points of different origins.

Raw events (including fallbacks) remain under `events/`. Authoritative callback
samples and absence reports update `streams/sent`; automatic fallbacks update
`bookkeeping/sent` separately. Without any callback, the sent row shows unavailable
with the fallback reason. Once callbacks exist, it shows the latest callback,
including authoritative absence reports and reasons; a fallback cannot hide it.
The panel separately identifies the no-evidence submission return by receipt
sequence, chunk/step and host time. A newer return does not relabel an older
callback as evidence for the new submission.

`StepRef` identifies a policy chunk and step, not an individual repeat execution.
No callback-to-submission attempt is inferred from event order or timestamps.
If the SDK can establish the exact session receipt, it may supply the optional
`submission_sequence` on its telemetry event; this is recorded and displayed as
provided. A synchronous callback before submit returns will normally know only
the chunk/step, and stays that way. Without per-attempt correlation, its value
remains independent of fallback bookkeeping even when the same step is submitted
again. Real adapter absence reports use the default `fallback=False`.

## Lifetime and queues

Accept validates every step before changing active chunk, selection or lifecycle
time. Chunk IDs are unique for one `PolicyInput` lifetime. Its small ID registry
is cleared on close; keep policy lifetimes bounded for indefinite services.
Input arrays and mappings are immutable copies, including preview references.
Supersession/expiry clears active selection and visual references. It does not
cancel a target in the robot SDK. Close releases this lease only and cannot
release a replacement lease. Existing Phase 2 ownership remains authoritative.

Only the bounded `VisualizationWorker` owns display consumption. Queue drops may
omit entire events; display is not a complete control audit. `stats` includes
accepted, consumed (including errored callbacks), pending, dropped_full,
discarded_close, rejected_closed, errors, last_error, closed and alive. A callback
runs outside the bookkeeping lock, so slow rendering cannot hold up publication
or target submission. Consumer errors are counted and the worker continues.

The Elite mock CLI additionally enables `coalesce_feedback=True`: newer
available feedback/planned display samples replace older pending samples of
the same origin, then are appended in host-time order. Control lifecycle, sent
evidence, absence and different-origin samples remain distinct. `stats.coalesced`
counts these replaced samples. The generic worker's default remains FIFO.
Elite displays robot, remaining reference, selected target and received TCP
trail in one shared 3D view; `--telemetry` adds only a text panel. By default,
the example consumes both arms from the bundled Elite dataset episode 0's `action`
column: 4451 synchronized pairs at 30 Hz, about 148.37 seconds, played once. Each 60-sample
chunk retains its original source time; selection uses the original timestamp
of each row, independently of display cadence. The 6D rotation contains the
first two **columns** of the world rotation matrix and is orthonormalized at
load time. XYZ values remain in metres without scaling or recentering.

`--arm both` is the default. Both policies share one input lease and SDK owner
thread. They select the same source frame and submit both targets before one
`StationSimulation.step()`, retaining the native dual-arm collision coordinator.
`--arm left` / `--arm right` selects a single arm; the other stays at startup.
The renderer keeps independent group chunks, selected points, receipts and
planned/sent telemetry under `.../groups/left` and `.../groups/right`, with one
robot model, one shared 3D view and one full-station feedback update. Stream
events carry an optional `group`; display coalescing only replaces samples of
the same stage **and group**, so a right-arm value cannot evict a left-arm value.
The dataset action is a `reference`, not a policy inference or newly measured
feedback. The robot/green trail and planned values still come from the native
simulation. Dataset observations are not substituted for simulation feedback;
Gripper actions (`<arm>_gripper.pos`, 0 closed to 1 open) are submitted through
the same input lease and at the same source time as the arm's TCP target, before
the station step. Each submission row retains its opening and separate gripper
receipt sequence. Recorded playback requires the selected arms' gripper columns.
The adapter maps opening through the station's calibration to independent
finger joints (`left_gripper` / `right_gripper` control groups); URDF mimic joints
follow them. A rate-limited offline model uses URDF joint velocity limits, and
the native collision checker certifies each finger segment before it advances.
An unsafe segment holds the fingers. The native arm collision coordinator and
renderer receive the same expanded finger configuration. This models opening
motion, not gripper motor dynamics or object contact forces.

The blue TCP geometry is trimmed from the selected index onward; the original
full chunk is still recorded as reference metadata. Orange marks the selected
**target** TCP, while the green endpoint marks the current **simulated** TCP.
These points are world XYZ of the configured tool frame, independent of jaw
opening, not one physical fingertip. Selected-point text labels are hidden.
Source metadata,
file hashes and the selected arm are recorded under `policy/.../reference_source`
and in the CLI result JSON. No remote mount, camera videos or LeRobot runtime
are needed. `scripts/import_elite_episode.py` can export another episode for
`--trajectory path.npz`.

`--duration N` limits playback seconds. `--source synthetic` retains the old
12-second example; `--kind joint` selects that source implicitly because the
dataset has no joint angles. `--no-viewer --fast --steps N` is the explicit
finite verification path. Both sources keep native planning/collision limits.
The host must inspect counters and the Boolean close result. A custom sink can
violate the nonblocking contract; do not pass synchronous rendering as the
publisher. Publisher exceptions are contained so an SDK-accepted command still
returns its submitted receipt; inspect `display_errors` / `display_drops`.

Close producers on the SDK owner thread, then drain the display worker with a
finite timeout. The example flushes/disconnects Rerun in `close_callback`, on the
display thread. That callback runs once after drain/discard. No background thread
gets an adapter, SDK model, native planner, input lease or actuator callback.

`replay_events(events, renderer.consume)` only iterates immutable data into a
display sink. `rmw-policy-replay recording.rrd` launches the Rerun Viewer without
constructing a simulation/controller. Neither mode executes recorded requests.
This package supplies no universal scheduler, IK, inference, stop/cancel API,
transport, safety validation or real-time guarantee.
