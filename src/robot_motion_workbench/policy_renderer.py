"""Read-only Rerun display of full policy references and associated streams."""
from dataclasses import asdict, dataclass, field, replace
import json

import numpy as np

from .control import JointTarget
from .policy import PolicyChunk, StepRef, StreamEvent
from .rerun_renderer import RerunRenderer, segment


def command_data(command):
    if command is None:
        return None
    if isinstance(command, JointTarget):
        return dict(group=command.group, positions=dict(command.positions))
    return dict(tcp=command.tcp, world_from_tcp=command.world_from_tcp.tolist())


def event_data(event):
    """JSON-compatible evidence; full denormalized predictions remain intact."""
    result = dict(stage=event.stage, timing=asdict(event.timing), origin=event.origin,
                  ref=asdict(event.ref) if event.ref else None, available=event.available,
                  reason=event.reason, command=command_data(event.command), fallback=event.fallback,
                  submission_sequence=event.submission_sequence)
    if event.group is not None:
        result['group'] = event.group
    if event.receipt:
        result['receipt'] = dict(sequence=event.receipt.sequence, source=event.receipt.source,
                                 status=event.receipt.status, command=command_data(event.receipt.command))
    if event.state:
        # RobotState.time is the adapter's rendering clock. It may be host
        # poll time; only SampleTime.source_time is an actual declared source.
        result['feedback'] = dict(state_time=event.state.time,
            joint_positions=dict(event.state.joint_positions),
            tcp_poses={k: v.tolist() for k, v in event.state.tcp_poses.items()})
    if event.chunk:
        c = event.chunk
        result['chunk'] = dict(chunk_id=c.chunk_id, group=c.group, joint_names=c.joint_names,
            time_offsets=c.time_offsets.tolist(), intended_host_times=(c.timing.received_at + c.time_offsets).tolist(),
            values=c.values.tolist(), timing=asdict(c.timing), expires_at=c.expires_at,
            kind=c.kind, frame=c.frame, tcp=c.tcp, origin=c.origin)
        result['preview'] = None if event.preview is None else dict(
            provider=event.preview.provider, joint_names=event.preview.joint_names,
            values=event.preview.values.tolist(), origin='reference')
    return result


@dataclass
class _PolicyChannel:
    root: str
    panel: str
    active: PolicyChunk | None = None
    selected: StepRef | None = None
    latest: dict[str, StreamEvent] = field(default_factory=dict)
    sent_fallback: StreamEvent | None = None
    status: str = 'No active chunk'
    preview_provider: str | None = None
    kind: str | None = None
    selected_points: np.ndarray | None = None
    offsets_text: str = ''


class PolicyRerunRenderer:
    """One display-thread consumer; it receives data, never a robot adapter.

    Predictions and ghost poses are all logged at chunk availability. Intended
    offsets are separate component values / reference plot X coordinates.
    Feedback is logged at host receipt, with source clock retained as metadata.
    Drop-newest display delivery may omit events: the status shows queue stats,
    and expiry is also checked against each received event's host time.
    Explicit groups keep independent reference/telemetry channels in one shared
    scene. Full-robot feedback is rendered once and updates every group's status.
    """
    def __init__(self, recording, description, *, timeline='sandbox_elapsed', max_age=0.25,
                 shared_view=False, telemetry=False, compact=False, groups=None):
        import rerun as rr
        self.rr, self.recording = rr, recording
        self.robot = RerunRenderer(recording, timeline=timeline, compact=compact)
        self.model = self.robot.add_robot(description)
        self.description, self.timeline, self.max_age = description, timeline, max_age
        self.root = f'world/policy/{segment(description.instance_id)}'
        self.panel = f'policy/{segment(description.instance_id)}'
        groups = tuple(groups) if groups is not None else (None,)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError('Policy display groups must be nonempty and unique')
        if len(groups) > 1 and not shared_view:
            raise ValueError('Multiple policy groups require shared_view=True')
        self._channels = {group: _PolicyChannel(
            self.root if group is None else f'{self.root}/groups/{segment(group)}',
            self.panel if group is None else f'{self.panel}/groups/{segment(group)}') for group in groups}
        self._rendered_feedback = None
        self._sequence = 0
        self._last_time = -1.0
        self._feedback_paths = {}
        self._shared_view, self._telemetry = shared_view, telemetry
        self._scene = None
        self._camera_initialized = False
        self.blueprint()

    def blueprint(self):
        import rerun.blueprint as rrb
        channel = next(iter(self._channels.values()))
        status = rrb.TextDocumentView(origin=f'{self.panel}/status', name='Provenance / selection / freshness')
        if self._shared_view:
            if self._scene is None:
                self._scene = rrb.Spatial3DView(origin='world',
                    contents=['/world/**', f'- /{self.robot.instance_path(self.description.instance_id)}/collision/**'],
                    name='TCP: remaining reference (blue) / target (orange) / simulated history (green)')
            contents = rrb.Horizontal(self._scene, status, column_shares=[4, 1]) if self._telemetry else self._scene
            self.recording.send_blueprint(rrb.Blueprint(contents, rrb.TimePanel(
                timeline=self.timeline,
                play_state='following' if not self._camera_initialized else None)))
            return
        tcp = segment(channel.active.tcp) if channel.active is not None and channel.active.tcp else '*'
        eye = None
        if channel.active is not None and channel.active.kind == 'absolute_tcp_poses':
            center = channel.active.values[:, :3, 3].mean(axis=0)
            extent = max(float(np.ptp(channel.active.values[:, :3, 3], axis=0).max()), 0.1)
            eye = rrb.EyeControls3D(position=center + extent * np.array([1.4, -1.8, 1.6]),
                                   look_target=center, eye_up=[0, 0, 1])
        reference = (rrb.Spatial2DView(origin=f'{channel.panel}/joint_reference',
                     name='Full joint chunk: offset (s) / SI position')
                     if channel.kind == 'absolute_joint_positions' else
                     rrb.Spatial3DView(origin='world', contents=[f'/{channel.root}/chunks/**',
                         f'/{self.root}/feedback_path/{tcp}',
                         f'/{self.robot.instance_path(self.description.instance_id)}/tcp/{tcp}/**'],
                         eye_controls=eye,
                         name='Future reference / received TCP close-up'))
        right = rrb.Vertical(reference, status, row_shares=[2, 3])
        self.recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin='world', contents=['/world/**',
                f'- /{self.robot.instance_path(self.description.instance_id)}/collision/**'],
                name='Current robot + full future reference'),
            right, column_shares=[3, 2])))

    def _clear(self, channel, chunk_id):
        self.recording.log(f'{channel.root}/chunks/{segment(chunk_id)}/prediction', self.rr.Clear(recursive=True))
        self.recording.log(f'{channel.panel}/joint_reference', self.rr.Clear(recursive=True))

    def _prediction(self, event, channel):
        c = event.chunk
        view_changed = channel.kind != c.kind or (channel.active is not None and channel.active.tcp != c.tcp)
        if channel.active is not None and channel.active.chunk_id != c.chunk_id:
            self._clear(channel, channel.active.chunk_id)
        channel.active, channel.selected = c, None
        channel.offsets_text = np.array2string(c.time_offsets, precision=2)
        channel.selected_points = c.values[:, :3, 3] if c.kind == 'absolute_tcp_poses' else None
        if view_changed:
            channel.kind = c.kind
            self.blueprint()
        channel.preview_provider = event.preview.provider if event.preview else None
        channel.status = f'Active {c.chunk_id}'
        path = f'{channel.root}/chunks/{segment(c.chunk_id)}/prediction'
        self.recording.log(path, self.rr.AnyValues(chunk_id=c.chunk_id, joint_names=list(c.joint_names),
            time_offsets=c.time_offsets, intended_host_times=c.timing.received_at + c.time_offsets,
            action_values=c.values.reshape(-1), action_shape=list(c.values.shape),
            origin=c.origin, frame=c.frame, semantics=c.kind))
        if c.kind == 'absolute_tcp_poses':
            points = c.values[:, :3, 3]
            self._tcp_path(path, points, c)
            for index, pose in enumerate(c.values):
                if self._shared_view and index not in (0, len(c.values) - 1):
                    continue
                axis = f'{path}/axes/{index}'
                self.robot._transform(axis, pose)
                self.recording.log(axis, self.rr.TransformAxes3D(0.025), self.rr.CoordinateFrame(self.robot.frame_id(axis)))
        else:
            for index, joint in enumerate(c.joint_names):
                self.recording.log(f'{channel.panel}/joint_reference/{segment(joint)}', self.rr.LineStrips2D(
                    [np.column_stack((c.time_offsets, c.values[:, index]))], labels=[joint], show_labels=True))
        if event.preview is not None:
            preview = event.preview
            tool_paths = {key: [] for key in self.description.tool_frames}
            for index, row in enumerate(preview.values):
                transforms = self.model.link_transforms(dict(zip(preview.joint_names, row, strict=True)), preview.base_transform)
                skeleton = [[transforms[j.parent][:3, 3], transforms[j.child][:3, 3]]
                            for j in self.model.ordered_joints]
                self.recording.log(f'{path}/ghosts/{index}', self.rr.LineStrips3D(skeleton,
                    colors=[130, 156, 255, 90], radii=0.002),
                    self.rr.CoordinateFrame(self.robot.frame_id('world')))
                for key, tool in self.description.tool_frames.items():
                    tool_paths[key].append((transforms[tool.parent_link] @ tool.link_from_tool)[:3, 3])
            for key, points in tool_paths.items():
                self._tcp_path(f'{path}/preview/{segment(key)}', np.asarray(points), c)
                if channel.selected_points is None or key == c.group:
                    channel.selected_points = np.asarray(points)
        if (self._shared_view and not self._camera_initialized and
                all(ch.selected_points is not None for ch in self._channels.values())):
            import rerun.blueprint as rrb
            # Frame the working arms and their short paths together, rather than
            # letting the large table/stand make the reference a few pixels.
            points = np.concatenate([ch.selected_points for ch in self._channels.values()])
            center = points.mean(axis=0)
            offset = np.array([.5, -.65, .45])
            if len(self._channels) > 1:
                offset *= max(1.5, float(np.ptp(points, axis=0).max()) / .4)
            self._scene.properties['EyeControls3D'] = rrb.EyeControls3D(
                position=center + offset, look_target=center + [0., 0., .07], eye_up=[0, 0, 1])
            self.blueprint()
            self._scene.properties.pop('EyeControls3D')
            self._camera_initialized = True

    def _tcp_path(self, path, points, chunk, start=0):
        points = points[start:]
        labels = [''] * len(points)
        for i in {0, len(labels) - 1}:
            labels[i] = f'{chunk.chunk_id}[{start + i}] +{chunk.time_offsets[start + i]:.2f}s reference'
        frame = self.rr.CoordinateFrame(self.robot.frame_id('world'))
        self.recording.log(f'{path}/tcp_path', self.rr.LineStrips3D([points],
            colors=[100, 164, 255], radii=0.006 if self._shared_view else 0.004), frame)
        self.recording.log(f'{path}/steps', self.rr.Points3D(points, colors=[100, 164, 255],
            radii=0.006, labels=labels, show_labels=not self._shared_view), frame)

    def consume(self, event: StreamEvent, *, stats=None):
        if None in self._channels:
            channel = self._channels[None]
        elif event.group is None and event.stage == 'feedback':
            channel = None  # One station snapshot covers all groups.
        else:
            try:
                channel = self._channels[event.group]
            except KeyError as exc:
                raise ValueError(f'Unknown policy display group: {event.group!r}') from exc
        now = event.timing.received_at
        if now < self._last_time:
            raise ValueError('Display host time moved backwards')
        self._last_time = now
        self.recording.set_time(self.timeline, duration=now)
        self._sequence += 1
        data = event_data(event)
        data['event_sequence'] = self._sequence
        document = self.rr.TextDocument(json.dumps(data, ensure_ascii=False), media_type='text/plain')
        # Keep event history in temporal rows. A new entity per event makes
        # long live recordings progressively slower in the Viewer.
        self.recording.log(f'{self.panel}/events/log', document)
        current_path = f'bookkeeping/{event.stage}' if event.fallback else f'streams/{event.stage}'
        panel = self.panel if channel is None else channel.panel
        self.recording.log(f'{panel}/{current_path}',
                          document)
        for ch in self._channels.values():
            if ch.active is not None and now >= ch.active.expires_at:
                self._clear(ch, ch.active.chunk_id)
                ch.status = f'Expired {ch.active.chunk_id}; previously submitted targets are SDK-owned'
                ch.active, ch.selected = None, None
        if event.stage == 'prediction':
            self._prediction(event, channel)
        elif event.stage in ('superseded', 'expired'):
            channel.status = f'{event.stage}: {event.ref.chunk_id}'
            if channel.active is not None and channel.active.chunk_id == event.ref.chunk_id:
                self._clear(channel, event.ref.chunk_id)
                channel.active, channel.selected = None, None
        elif event.stage == 'selection':
            channel.selected = event.ref
            if (channel.active is not None and channel.active.chunk_id == event.ref.chunk_id
                    and channel.selected_points is not None):
                if channel.active.kind == 'absolute_tcp_poses':
                    self._tcp_path(f'{channel.root}/chunks/{segment(event.ref.chunk_id)}/prediction',
                                   channel.selected_points, channel.active, start=event.ref.step_index)
                self.recording.log(f'{channel.root}/chunks/{segment(event.ref.chunk_id)}/prediction/selected',
                    self.rr.Points3D([channel.selected_points[event.ref.step_index]],
                        colors=[255, 187, 65], radii=0.008, show_labels=False),
                    self.rr.CoordinateFrame(self.robot.frame_id('world')))
        if event.stage in ('submitted', 'planned', 'sent', 'feedback'):
            for ch in self._channels.values() if channel is None else (channel,):
                if event.fallback:
                    ch.sent_fallback = event
                else:
                    ch.latest[event.stage] = event
            if event.state is not None:
                if self._rendered_feedback is not None and self._rendered_feedback.origin != event.origin:
                    # Never connect simulated and measured points into a path
                    # whose current legend would relabel the older segment.
                    self._feedback_paths.clear()
                    self.recording.log(f'{self.root}/feedback_path', self.rr.Clear(recursive=True))
                # State.time is the SDK's source clock, not a future recording
                # timestamp. Log current geometry at host receipt instead.
                self.robot.update(self.description.instance_id, replace(event.state, time=now))
                self._rendered_feedback = event
                self.recording.log(f'{self.panel}/rendered_feedback', document)
                for key, tcp in event.state.tcp_poses.items():
                    history = self._feedback_paths.setdefault(key, [])
                    history.append(tcp[:3, 3])
                    del history[:-2048]
                    self.recording.log(f'{self.root}/feedback_path/{segment(key)}', self.rr.LineStrips3D(
                        [history], colors=[65, 240, 120] if self._shared_view else [65, 210, 150],
                        radii=0.004 if self._shared_view else 0.003),
                        self.rr.CoordinateFrame(self.robot.frame_id('world')))
                    self.recording.log(f'{self.root}/feedback_path/{segment(key)}/current',
                        self.rr.Points3D([tcp[:3, 3]], colors=[65, 240, 120], radii=0.006, show_labels=False),
                        self.rr.CoordinateFrame(self.robot.frame_id('world')))
        # In the shared live view, update text alongside the pose/chunk rather
        # than rebuilding both arms' tables for every submission and fallback.
        if not self._shared_view or event.stage in ('feedback', 'prediction', 'superseded', 'expired'):
            self._draw_status(now, stats)

    def _draw_status(self, now, stats):
        feedback = self._rendered_feedback
        feedback_origin = feedback.origin if feedback is not None else 'unavailable'
        lines = ['## Policy references / received state',
                 '', 'Blue: remaining TCP reference in the current chunk. Orange: selected TCP target, not actual position.',
                 f'Green trail / endpoint / robot: received **{feedback_origin}** feedback (TCP history / current state).']
        if feedback is not None:
            age = now - feedback.timing.received_at
            lines += ['', f'Rendered state: host {feedback.timing.received_at:.3f}s; '
                      f'source {self._source(feedback.timing)}; age {age:.3f}s '
                      f'({feedback.timing.freshness(now, self.max_age)}).']
        for group, channel in self._channels.items():
            if group is not None:
                lines += ['', f'### {group}']
            lines += self._channel_status(channel, now)
        lines += ['', 'Setter success = submission only. Sent unavailable without send telemetry.',
                  '', 'Freshness = host receipt age. SDK owns motion; display/replay never calls control.']
        if stats is not None:
            lines += ['', f'Display: dropped={stats.dropped_full}, pending={stats.pending}, errors={stats.errors} (best effort).']
        self.recording.log(f'{self.panel}/status', self.rr.TextDocument('\n'.join(lines), media_type='text/markdown'))

    def _channel_status(self, channel, now):
        lines = ['', channel.status]
        if channel.active is not None:
            c = channel.active
            lines += ['', f'**{c.chunk_id}** · {c.kind} · {len(c.values)} steps · {c.origin}',
                      '', f'Host available {c.timing.received_at:.3f}s · expiry {c.expires_at:.3f}s · selected **{channel.selected.step_index if channel.selected else "none"}**',
                      '', f'Offsets {channel.offsets_text} s · source {self._source(c.timing)}',
                      '', f'Joint ghosts: {channel.preview_provider}' if channel.preview_provider
                      else 'Joint preview unavailable; no inferred IK.']
        lines += ['', '| Stream | Origin / freshness | Association |', '|---|---|---|']
        details = []
        for stage in ('submitted', 'planned', 'sent', 'feedback'):
            event = channel.latest.get(stage)
            if stage == 'sent' and event is None:
                event = channel.sent_fallback
            if event is None:
                lines.append(f'| {stage} | unavailable | none |')
                continue
            age = now - event.timing.received_at
            freshness = event.timing.freshness(now, self.max_age) if event.available else 'unavailable'
            association = (f'{event.ref.chunk_id}[{event.ref.step_index}]' if event.ref.step_index is not None
                           else f'{event.ref.chunk_id} (no selection)') if event.ref else 'unassociated'
            lines.append(f'| {stage} | {event.origin} / {freshness} | {association} |')
            command = event.receipt.command if event.receipt else event.command
            value = ''
            if command is not None:
                value = (f'{next(iter(command.positions))}={next(iter(command.positions.values())):.4f} SI'
                         if isinstance(command, JointTarget) else f'TCP xyz={np.round(command.world_from_tcp[:3, 3], 4)} m')
            if event.state is not None:
                value = f'{next(iter(event.state.joint_positions))}={next(iter(event.state.joint_positions.values())):.4f} SI'
            if event.submission_sequence is not None and not event.fallback:
                value += f' (session receipt #{event.submission_sequence})'
            details.append(f'- **{stage}** {value}; host {event.timing.received_at:.3f}s; source {self._source(event.timing)}; age {age:.3f}s')
            if event.reason and (not event.available or stage == 'sent'):
                details.append(f'  Reason: {event.reason}')
        lines += ['', *details]
        if channel.sent_fallback is not None:
            fallback = channel.sent_fallback
            sequence = f' #{fallback.submission_sequence}' if fallback.submission_sequence is not None else ''
            lines += ['', f'Submission return{sequence} at host {fallback.timing.received_at:.3f}s '
                      f'for {fallback.ref.chunk_id}[{fallback.ref.step_index}] supplies no send evidence '
                      '(fallback bookkeeping). Callback telemetry above is independent; '
                      'a chunk/step association alone does not identify a repeated submission.']
        return lines

    @staticmethod
    def _source(timing):
        return 'unknown' if timing.source_time is None else f'{timing.source_time:.3f}s ({timing.source_clock})'

    def close(self):
        self.robot.flush()
        self.recording.disconnect()
