"""Read-only Rerun display of full policy references and associated streams."""
from dataclasses import asdict, replace
import json

import numpy as np

from .control import JointTarget
from .policy import StreamEvent
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


class PolicyRerunRenderer:
    """One display-thread consumer; it receives data, never a robot adapter.

    Predictions and ghost poses are all logged at chunk availability. Intended
    offsets are separate component values / reference plot X coordinates.
    Feedback is logged at host receipt, with source clock retained as metadata.
    Drop-newest display delivery may omit events: the status shows queue stats,
    and expiry is also checked against each received event's host time.
    """
    def __init__(self, recording, description, *, timeline='sandbox_elapsed', max_age=0.25):
        import rerun as rr
        self.rr, self.recording = rr, recording
        self.robot = RerunRenderer(recording, timeline=timeline)
        self.model = self.robot.add_robot(description)
        self.description, self.timeline, self.max_age = description, timeline, max_age
        self.root = f'world/policy/{segment(description.instance_id)}'
        self.panel = f'policy/{segment(description.instance_id)}'
        self._active = None
        self._selected = None
        self._latest = {}
        self._rendered_feedback = None
        self._sent_fallback = None
        self._sequence = 0
        self._last_time = -1.0
        self._status = 'No active chunk'
        self._feedback_paths = {}
        self._preview_provider = None
        self._kind = None
        self.blueprint()

    def blueprint(self):
        import rerun.blueprint as rrb
        status = rrb.TextDocumentView(origin=f'{self.panel}/status', name='Provenance / selection / freshness')
        tcp = segment(self._active.tcp) if self._active is not None and self._active.tcp else '*'
        eye = None
        if self._active is not None and self._active.kind == 'absolute_tcp_poses':
            center = self._active.values[:, :3, 3].mean(axis=0)
            extent = max(float(np.ptp(self._active.values[:, :3, 3], axis=0).max()), 0.1)
            eye = rrb.EyeControls3D(position=center + extent * np.array([1.4, -1.8, 1.6]),
                                   look_target=center, eye_up=[0, 0, 1])
        reference = (rrb.Spatial2DView(origin=f'{self.panel}/joint_reference',
                     name='Full joint chunk: offset (s) / SI position')
                     if self._kind == 'absolute_joint_positions' else
                     rrb.Spatial3DView(origin='world', contents=[f'/{self.root}/chunks/**',
                         f'/{self.root}/feedback_path/{tcp}',
                         f'/{self.robot.instance_path(self.description.instance_id)}/tcp/{tcp}/**'],
                         eye_controls=eye,
                         name='Future reference / received TCP close-up'))
        right = rrb.Vertical(reference, status, row_shares=[2, 3])
        self.recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin='world', contents=['/world/**',
                f'- /{self.robot.instance_path(self.description.instance_id)}/collision/**'],
                name='Current robot + full future reference'),
            right, column_shares=[3, 2]), collapse_panels=True))

    def _clear(self, chunk_id):
        self.recording.log(f'{self.root}/chunks/{segment(chunk_id)}/prediction', self.rr.Clear(recursive=True))
        self.recording.log(f'{self.panel}/joint_reference', self.rr.Clear(recursive=True))

    def _prediction(self, event):
        c = event.chunk
        view_changed = self._kind != c.kind or (self._active is not None and self._active.tcp != c.tcp)
        if self._active is not None and self._active.chunk_id != c.chunk_id:
            self._clear(self._active.chunk_id)
        self._active, self._selected = c, None
        if view_changed:
            self._kind = c.kind
            self.blueprint()
        self._preview_provider = event.preview.provider if event.preview else None
        self._status = f'Active {c.chunk_id}'
        path = f'{self.root}/chunks/{segment(c.chunk_id)}/prediction'
        self.recording.log(path, self.rr.AnyValues(chunk_id=c.chunk_id, joint_names=list(c.joint_names),
            time_offsets=c.time_offsets, intended_host_times=c.timing.received_at + c.time_offsets,
            action_values=c.values.reshape(-1), action_shape=list(c.values.shape),
            origin=c.origin, frame=c.frame, semantics=c.kind))
        if c.kind == 'absolute_tcp_poses':
            points = c.values[:, :3, 3]
            self._tcp_path(path, points, c)
            for index, pose in enumerate(c.values):
                axis = f'{path}/axes/{index}'
                self.robot._transform(axis, pose)
                self.recording.log(axis, self.rr.TransformAxes3D(0.025), self.rr.CoordinateFrame(self.robot.frame_id(axis)))
        else:
            for index, joint in enumerate(c.joint_names):
                self.recording.log(f'{self.panel}/joint_reference/{segment(joint)}', self.rr.LineStrips2D(
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

    def _tcp_path(self, path, points, chunk):
        labels = [''] * len(chunk.time_offsets)
        for i in {0, len(labels) - 1}:
            labels[i] = f'{chunk.chunk_id}[{i}] +{chunk.time_offsets[i]:.2f}s reference'
        frame = self.rr.CoordinateFrame(self.robot.frame_id('world'))
        self.recording.log(f'{path}/tcp_path', self.rr.LineStrips3D([points],
            colors=[100, 164, 255], radii=0.004), frame)
        self.recording.log(f'{path}/steps', self.rr.Points3D(points, colors=[100, 164, 255],
            radii=0.006, labels=labels, show_labels=True), frame)

    def consume(self, event: StreamEvent, *, stats=None):
        now = event.timing.received_at
        if now < self._last_time:
            raise ValueError('Display host time moved backwards')
        self._last_time = now
        self.recording.set_time(self.timeline, duration=now)
        self._sequence += 1
        data = event_data(event)
        self.recording.log(f'{self.panel}/events/{self._sequence:08d}',
                          self.rr.TextDocument(json.dumps(data, ensure_ascii=False), media_type='text/plain'))
        current_path = f'bookkeeping/{event.stage}' if event.fallback else f'streams/{event.stage}'
        self.recording.log(f'{self.panel}/{current_path}',
                          self.rr.TextDocument(json.dumps(data, ensure_ascii=False), media_type='text/plain'))
        if self._active is not None and now >= self._active.expires_at:
            self._clear(self._active.chunk_id)
            self._status = f'Expired {self._active.chunk_id}; previously submitted targets are SDK-owned'
            self._active, self._selected = None, None
        if event.stage == 'prediction':
            self._prediction(event)
        elif event.stage in ('superseded', 'expired'):
            self._status = f'{event.stage}: {event.ref.chunk_id}'
            if self._active is not None and self._active.chunk_id == event.ref.chunk_id:
                self._clear(event.ref.chunk_id)
                self._active, self._selected = None, None
        elif event.stage == 'selection':
            self._selected = event.ref
            if (self._active is not None and self._active.chunk_id == event.ref.chunk_id
                    and self._active.kind == 'absolute_tcp_poses'):
                self.recording.log(f'{self.root}/chunks/{segment(event.ref.chunk_id)}/prediction/selected',
                    self.rr.Points3D([self._active.values[event.ref.step_index, :3, 3]],
                        colors=[255, 187, 65], radii=0.008,
                        labels=[f'selected [{event.ref.step_index}]'], show_labels=True),
                    self.rr.CoordinateFrame(self.robot.frame_id('world')))
        if event.stage in ('submitted', 'planned', 'sent', 'feedback'):
            if event.fallback:
                self._sent_fallback = event
            else:
                self._latest[event.stage] = event
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
                self.recording.log(f'{self.panel}/rendered_feedback', self.rr.TextDocument(
                    json.dumps(data, ensure_ascii=False), media_type='text/plain'))
                for key, tcp in event.state.tcp_poses.items():
                    history = self._feedback_paths.setdefault(key, [])
                    history.append(tcp[:3, 3])
                    del history[:-2048]
                    self.recording.log(f'{self.root}/feedback_path/{segment(key)}', self.rr.LineStrips3D(
                        [history], colors=[65, 210, 150], radii=0.003),
                        self.rr.CoordinateFrame(self.robot.frame_id('world')))
        self._draw_status(now, stats)

    def _draw_status(self, now, stats):
        feedback = self._rendered_feedback
        feedback_origin = feedback.origin if feedback is not None else 'unavailable'
        lines = ['## Policy references / received state', '', self._status,
                 '', f'Blue: future reference. Green / robot: received **{feedback_origin}** feedback.']
        if feedback is not None:
            age = now - feedback.timing.received_at
            lines += ['', f'Rendered state: host {feedback.timing.received_at:.3f}s; '
                      f'source {self._source(feedback.timing)}; age {age:.3f}s '
                      f'({feedback.timing.freshness(now, self.max_age)}).']
        if self._active is not None:
            c = self._active
            lines += ['', f'**{c.chunk_id}** · {c.kind} · {len(c.values)} steps · {c.origin}',
                      '', f'Host available {c.timing.received_at:.3f}s · expiry {c.expires_at:.3f}s · selected **{self._selected.step_index if self._selected else "none"}**',
                      '', f'Offsets {np.array2string(c.time_offsets, precision=2)} s · source {self._source(c.timing)}',
                      '', f'Joint ghosts: {self._preview_provider}' if self._preview_provider
                      else 'Joint preview unavailable; no inferred IK.']
        lines += ['', '| Stream | Origin / freshness | Association |', '|---|---|---|']
        details = []
        for stage in ('submitted', 'planned', 'sent', 'feedback'):
            event = self._latest.get(stage)
            if stage == 'sent' and event is None:
                event = self._sent_fallback
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
        if self._sent_fallback is not None:
            fallback = self._sent_fallback
            sequence = f' #{fallback.submission_sequence}' if fallback.submission_sequence is not None else ''
            lines += ['', f'Submission return{sequence} at host {fallback.timing.received_at:.3f}s '
                      f'for {fallback.ref.chunk_id}[{fallback.ref.step_index}] supplies no send evidence '
                      '(fallback bookkeeping). Callback telemetry above is independent; '
                      'a chunk/step association alone does not identify a repeated submission.']
        lines += ['', 'Setter success = submission only. Sent unavailable without send telemetry.',
                  '', 'Freshness = host receipt age. SDK owns motion; display/replay never calls control.']
        if stats is not None:
            lines += ['', f'Display: dropped={stats.dropped_full}, pending={stats.pending}, errors={stats.errors} (best effort).']
        self.recording.log(f'{self.panel}/status', self.rr.TextDocument('\n'.join(lines), media_type='text/markdown'))

    @staticmethod
    def _source(timing):
        return 'unknown' if timing.source_time is None else f'{timing.source_time:.3f}s ({timing.source_clock})'

    def close(self):
        self.robot.flush()
        self.recording.disconnect()
