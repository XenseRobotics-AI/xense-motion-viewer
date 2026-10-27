# Public adapter recipe

The SDK owns joint count/order, IK, planning, limits, collision checks, and any
driver thread. The workbench submits named targets and reads snapshots. The
installed `rmw-callback-demo` is a complete runnable example with the packaged
original URDF; it needs only NumPy and opens no hardware. Its bridge is in
[`examples/callback_sdk.py`](../src/robot_motion_workbench/examples/callback_sdk.py).

```python
from robot_motion_workbench import JointTarget
from robot_motion_workbench.adapters.reference import description_from_urdf, urdf_joint_controls
from robot_motion_workbench.examples.callback_sdk import make_callback_session

description = description_from_urdf()  # Packaged original primitive model.
# Production: supply your SDK's verified limits and calibrated description.
controls = urdf_joint_controls(description)
sdk_order = tuple(reversed(description.joint_names))
observed = [0.0] * len(sdk_order)
accepted = []
session = make_callback_session(
    description, controls, sdk_joint_names=sdk_order,
    read_positions=lambda: observed.copy(), write_positions=accepted.append,
)
inputs = session.activate('application')
goal = {n: 0.01 * (i + 1) for i, n in enumerate(description.joint_names)}
receipt = inputs.submit(JointTarget('arm', goal))
assert receipt.status == 'submitted'
assert accepted == [[goal[n] for n in sdk_order]]
assert all(v == 0.0 for v in session.snapshot().joint_positions.values())
inputs.close()
```

Replace the two callbacks with methods on an already configured caller-owned
SDK object. Pass an explicit URDF / `RobotDescription` and SDK vector order;
no name guessing, fixed DOF, connection or control lifecycle is hidden here.
`read_positions` returns one finite position per SDK-ordered joint in SI;
`write_positions` accepts that complete SDK vector and may raise. Poll time
is elapsed host receipt, **not** a source sample timestamp. Actual-send,
source time, calibrated TCP and planner telemetry remain absent unless the SDK
supplies independent evidence.

For TCP control construct `CallbackRobotAdapter` directly with a
`ControlCapabilities.tcp_targets` mapping and callbacks accepting `TcpTarget`;
the adapter owns world/base/tool conversion and SDK IK. Omit that mapping when
no Cartesian controller exists. The reference and Flexiv joint adapters omit it.

For a class-based bridge implement `describe() -> RobotDescription`,
`capabilities() -> ControlCapabilities`, `snapshot() -> RobotState`, and
`submit(JointTarget | TcpTarget) -> None`. The public
`ControllableRobotAdapter` protocol requires no inheritance. Visualization-only
adapters can implement just `RobotAdapter.describe()` / `snapshot()`.

The optional `adapters.flexiv.FlexivJointAdapter` is a seven-axis vendor mapping,
not a control implementation. It receives a caller-owned already configured
joint controller and maps only `is_running`, `get_state().q` and
`set_target_joints`. The verified API surface and its missing send/source-time
telemetry are described in the [README](../README.md#connect-an-external-sdk).
Its seven-axis constraint belongs to this vendor adapter; the core, Qt and
policy have none. No vendor import is required to use or inspect the bridge.

## Commands, metadata and receipts

`JointTarget(group, positions)` must supply exactly the named joints in that
group. Groups may overlap, expose a subset, or have an order different from the
description. A group target has no effect on other groups unless the SDK itself
defines coupled behavior. Unknown, missing, non-finite and statically out-of-limit
values fail before `adapter.submit`; values are never clipped or padded. Joint
values are radians or metres according to `JointControl.unit`. Optional bounds
come from the SDK; a continuous joint may omit them. UI conversion is
`display = SI * display_scale + display_offset` and can use a negative scale.
Metadata is copied and frozen for the session's lifetime; model/capability changes
require a new session.

`TcpTarget(tcp, world_from_tcp)` accepts only a finite, right-handed rigid 4×4
transform. Translation uses metres. The only command frame in this version is
world; the adapter handles base/tool conversion. TCP names must appear in
`capabilities.tcp_targets`, where each TCP references a known control group.
Remove that mapping to disable Cartesian commands and UI. A valid pose is not
a guarantee of reachability; IK and dynamic target checks remain SDK-owned.

Commands, arrays, mappings, metadata, and receipts are immutable copies. The
session validates before invoking the adapter; SDK rejection raises and creates
no new receipt. After callback return it allocates a session-local increasing
sequence and records `last_receipt`. The receipt is solely **submitted** and
does not claim a hardware send, planner acceptance at a later tick, motion,
tracking, or achievement. Use independent SDK feedback in `snapshot()`; never
derive a measured joint position or TCP pose from the receipt. Elite snapshots
are simulation observations, not hardware measurements.

## Producer ownership and Qt

```python
from PySide6.QtWidgets import QApplication
from robot_motion_workbench.qt import ControlPanel

app = QApplication.instance() or QApplication([])
qt_input = session.activate("qt")
panel = ControlPanel(qt_input)
panel.show()
# The host may call panel.refresh() from its own GUI observation timer.
# It never submits or advances a robot. Only Submit buttons produce targets.
app.exec()
qt_input.close()
```

`rmw-demo-qt` opens the input panel; `--save` records an RRD without launching
or connecting a live Rerun Viewer. Close the panel to finish recording, then
use `rmw-policy-replay saved-path.rrd`. Elite's `rmw-elite-qt` retains its
separate live Rerun Viewer unless `--no-viewer` is selected.

For application-owned live visualization, configure a connection on a
caller-owned `rerun.RecordingStream` and pass it to the public
`robot_motion_workbench.rerun_renderer.RerunRenderer`. The caller owns the
stream/viewer lifecycle, rendering thread and SDK thread marshalling.

The panel builds arbitrary joint/group fields from capabilities and adds
world-XYZ/fixed-axis-RPY editing only for advertised TCP targets. It displays
observations separately from target drafts. `refresh()` changes observation
labels only; `sync_targets()` explicitly copies observations into drafts with
signals blocked and still sends nothing. A TCP without an observation has no
editable default pose; explicitly sync after an observation becomes available.
UI ranges are editing aids, not reachability claims. Core commands retain full
precision; GUI numeric editors have finite display precision.

There is one active producer per robot session. `session.activate("policy")`
fails while Qt owns it; the host explicitly transfers with
`session.activate("policy", replace=True)`. Old sessions remain invalid even
if their source name is reused, and closing an old session cannot release its
replacement. Closing ownership does **not** stop the robot or cancel its last
target. There is no automatic idle takeover. Qt refresh disables generic submit
buttons after takeover; dispatch also rejects stale input even before refresh.

Calls are synchronous and serialized across session submit/snapshot/ownership
methods. No worker, queue, TTL, universal update frequency, IK, retry, motion
interpolation, or real-time guarantee is supplied. An adapter must handle SDK
thread affinity and synchronize with any SDK-owned loop; an application must
not call Elite's offline runtime concurrently from multiple host threads.

## Elite host and compatibility

`EliteStationAdapter` constructs no driver and opens no socket. It exposes one
named group/TCP per simulator arm, maps named group positions into the SDK's
order, and calls the existing `StationSimulation.set_joint_target` or
`set_target`. The latter retains the production world-to-base conversion.
`EliteSandboxRuntime` owns offline elapsed-time catch-up and invokes unchanged
`StationSimulation.step`, including the production planner, sender gate,
collision guard, latching, recipe TCP and dynamic target-jump rules.

The Elite adapter additionally exposes two independent finger joints and
`left_gripper` / `right_gripper` groups. `adapter.grippers.target(side, opening)`
converts normalized opening (0 closed, 1 open) through the station calibration
to an ordinary SI `JointTarget`, submitted through the same session. Before
each native arm step, the offline finger model advances within URDF velocity
limits and checks the full finger segment with the native collision checker.
Unsafe motion holds. Mimic expansion, native guard configuration and rendered
snapshots agree on the finger positions. This is a kinematic opening simulation,
not motor dynamics or grasp/contact simulation; it opens no gripper connection.

`rmw-elite-qt` preserves the useful original Elite slider/FK/IK workflow via a
thin facade and readable original classification diagnostics. This vendor UI
is specifically for Elite; the generic panel makes no six-axis assumptions.
Both UIs and `rmw-elite-record --jog-m ...` submit through `RobotSession`.
Both Elite Qt layouts place each gripper slider below its arm's six joint
controls and below yaw in the IK page. Both copies share target values and
separate observed opening labels; synchronizing copies never submits a command. A deliberate gripper
edit also requests the current simulation time and playback in the live Viewer
on the display worker, preserving its camera and panels; idle observation does
not seek the Viewer. Observation refresh blocks submission; ownership takeover
disables these controls, and stale signals are rejected by the same session.
`rmw-elite-qt --ui generic` selects the metadata panel. Explicit home/reset
controls remain offline simulation lifecycle operations, require Qt ownership,
and are not portable robot-motion commands. Rendering preserves its monotonic
output clock across reset without changing planner clocks.

Elite Qt delivers copied snapshots through a 30 Hz latest-state mailbox: one
frame may be in flight and one pending frame is replaced by newer state.
Collision visibility uses that same display thread and keeps the view ID.
Compact rendering uses relative joint transforms and exact collision-mesh
concatenation by parent/color; the native control/collision path and poll
scheduling are unchanged. Close waits at most one second and exposes timeouts
and display errors. The Elite CLI configures file + gRPC sinks together before
logging, with a 33 ms batch tick; it does not replace a file sink by spawning.

The optional `elite` extra alone cannot guarantee the current planner: use the
matching wheel in `vendor/wheels/`, with its source revision and hashes recorded
in [the SDK provenance](../vendor/libpyelite-provenance.json), rather than trusting
its version number. The SDK (including native code and URDF/mesh assets) is
installed in the Python environment, not loaded from a repository submodule.
Missing Elite imports produce installation guidance with
the original exception as their cause. No hardware integration is validated
by these offline examples or tests.
