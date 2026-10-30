"""Qt-only independent virtual hands; all robot motion stays in the native SDK."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget, QSizePolicy,
)

from ..follow import FollowInput, FollowTrace, command_data, command_from_data, rotation_vector
from ..control import TcpTarget


class ManualFollowInput:
    """Manual finger edits cannot mix into playback or a faulted simulation."""
    def __init__(self, panel):
        self.panel = panel

    @property
    def active(self):
        c = self.panel.controller
        return c.active and c.playback is None

    def submit(self, command):
        if not self.active:
            raise ValueError('手动夹爪输入未启用，或正在回放 / 暂停')
        return self.panel.inputs.submit(command)


def build_info(simulation):
    import libpyelite
    from libpyelite.control_profile import describe_config
    native = Path(libpyelite._native.__file__)
    configs = {}
    for side, cfg in simulation.configs.items():
        configs[side] = describe_config(cfg)
        path = Path(configs[side].pop('model_path'))
        configs[side]['model_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    environment = dict(configs=configs, home_q=simulation.home_q.tolist(),
        station_urdf_sha256=hashlib.sha256(simulation.urdf_path.read_bytes()).hexdigest(),
        collision_profile=simulation.collision.profile if simulation.collision else None)
    signature = hashlib.sha256(json.dumps(environment, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(native_path=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                sdk_version=libpyelite.__version__, environment=environment, environment_sha256=signature,
                scope='offline ideal plant; viewer grippers use collision preflight')


def feedback(simulation, samples):
    def number(value):
        return float(value) if np.isfinite(value) else None
    return dict(q=simulation.q.tolist(), tcp={s: p.tolist() for s, p in simulation.tcp_poses().items()},
        reference_q={s: list(a.planner.reference()) for s, a in simulation.arms.items()},
        collision={s: dict(held=bool(v.get('collision_hold', False)), limited=bool(v.get('collision_limited', False)),
                          distance_m=number(v.get('collision_distance_m', float('inf')))) for s, v in samples.items()})


def repro_trace(simulation, metadata, *, side='left', angle=.08):
    """The deterministic collision regression, with explicit offline start pose."""
    if side not in simulation.arms or not np.isfinite(angle):
        raise ValueError('Invalid reproduction parameters')
    if simulation.assembly.root.name != 'diagonal-08':
        raise ValueError('这个离线反例使用 diagonal-08，请使用对应站点')
    q = simulation.home_q.copy()
    initial = np.array([-1.15, -.53, 1.45, -.62, 1.78, -.66])
    if side == 'right':
        initial[[0, 1, 3]] = -np.pi - initial[[0, 1, 3]]
        initial[[2, 4]] *= -1
    index = list(simulation.arms).index(side)
    q[index * 6:(index + 1) * 6] = initial
    arm = simulation.arms[side]
    home = simulation.assembly.world_from_base(side) @ arm.model.fk(initial)
    ticks = max(1, round(.032 / arm.config.period))
    hands = {s: np.eye(4).tolist() for s in simulation.arms}
    trace = FollowTrace(metadata, q.tolist(), hands, arm.config.period, duration_steps=350 * ticks,
                        end_reason='preset: approach 0.12m + rotation, hold, retreat')
    for frame in range(350):
        fraction = min(frame / 30, 1.) if frame < 220 else max(0., 1. - (frame - 220) / 30)
        delta = np.eye(4)
        delta[2, 3] = -.12 * fraction
        delta[:3, :3] = rotation_vector([0., angle * fraction, 0.])
        pose = home.copy()
        pose[:3, 3] += delta[:3, 3]
        pose[:3, :3] = delta[:3, :3] @ home[:3, :3]
        current_hands = dict(hands) | {side: delta.tolist()}
        trace.commands.append(dict(step=frame * ticks, command=command_data(TcpTarget(side, pose)), hands=current_hands))
    trace.validate()
    return trace


class HandPad(QWidget):
    """A virtual hand in a selectable world plane; feedback never moves it."""
    PLANES = {'XY': (0, 1, 2), 'XZ': (0, 2, 1), 'YZ': (1, 2, 0)}

    def __init__(self, controller, side, report, parent=None):
        super().__init__(parent)
        self.controller, self.side, self.report = controller, side, report
        self.plane, self.rotation_axis = 'XY', 1
        self.metres_per_pixel = .001
        self.centre = np.zeros(3)
        self.buttons = set()
        self.keys = set()
        self.last_mouse = QPointF()
        self.setMinimumSize(330, 210)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setObjectName(side + '_follow_hand')
        self.setToolTip('左键拖球平移；滚轮调节第三轴；右键拖动旋转；Q/E 持续旋转，可与平移同时进行')

    def point(self):
        a, b, _ = self.PLANES[self.plane]
        position = self.controller.hands[self.side][:3, 3] - self.centre
        return QPointF(self.width()/2 + position[a]/self.metres_per_pixel,
                       self.height()/2 - position[b]/self.metres_per_pixel)

    def recenter(self):
        self.centre = self.controller.hands[self.side][:3, 3].copy()
        self.update()

    def _try(self, callback):
        try:
            callback()
            self.update()
            return True
        except Exception as exc:
            self.report(str(exc))
            return False

    def release_input(self):
        self.buttons.clear()
        self.keys.clear()
        self.controller.end_grip(self.side)
        self.update()

    def mousePressEvent(self, event):
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if (event.position()-self.point()).manhattanLength() > 36:
            return
        self.setFocus()
        if self._try(lambda: self.controller.begin_grip(self.side)):
            self.buttons.add(event.button())
            self.last_mouse = event.position()
            event.accept()

    def mouseMoveEvent(self, event):
        if not self.buttons:
            return
        delta = event.position() - self.last_mouse
        self.last_mouse = event.position()
        translation, rotation = np.zeros(3), np.zeros(3)
        if Qt.MouseButton.LeftButton in self.buttons:
            a, b, _ = self.PLANES[self.plane]
            translation[a], translation[b] = delta.x()*self.metres_per_pixel, -delta.y()*self.metres_per_pixel
        if Qt.MouseButton.RightButton in self.buttons:
            rotation[self.rotation_axis] = delta.x()*.004
        self._try(lambda: self.controller.move_hand(self.side, translation, rotation))
        point = self.point()
        if not (20 < point.x() < self.width()-20 and 20 < point.y() < self.height()-20):
            self.recenter()  # Camera only; never clamp or reset the hand/robot.

    def mouseReleaseEvent(self, event):
        self.buttons.discard(event.button())
        if not self.buttons and not self.keys:
            self.controller.end_grip(self.side)
        self.update()

    def wheelEvent(self, event):
        already_gripped = self.side in self.controller.anchors
        if not self._try(lambda: self.controller.begin_grip(self.side)):
            return
        delta = np.zeros(3)
        delta[self.PLANES[self.plane][2]] = event.angleDelta().y()/120*.005
        self._try(lambda: self.controller.move_hand(self.side, delta))
        if not already_gripped:
            self.controller.end_grip(self.side)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() not in (Qt.Key.Key_Q, Qt.Key.Key_E) or event.isAutoRepeat():
            return
        if self._try(lambda: self.controller.begin_grip(self.side)):
            self.keys.add(event.key())
            self._rates()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        self.keys.discard(event.key())
        self._rates()
        if not self.keys and not self.buttons:
            self.controller.end_grip(self.side)

    def _rates(self):
        rate = self.controller.angular_rates[self.side]
        rate.fill(0)
        rate[self.rotation_axis] = .3 * ((Qt.Key.Key_E in self.keys) - (Qt.Key.Key_Q in self.keys))

    def focusOutEvent(self, event):
        self.release_input()
        super().focusOutEvent(event)

    def hideEvent(self, event):
        self.release_input()
        super().hideEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#101b2b'))
        p.setPen(QPen(QColor('#223349'), 1))
        for x in range(int(self.width()/2)%40, self.width(), 40):
            p.drawLine(x, 0, x, self.height())
        for y in range(int(self.height()/2)%40, self.height(), 40):
            p.drawLine(0, y, self.width(), y)
        point = self.point()
        gradient = QRadialGradient(point-QPointF(5, 5), 24)
        gradient.setColorAt(0, QColor('#d4f1ff'))
        gradient.setColorAt(.45, QColor('#60b9e8'))
        gradient.setColorAt(1, QColor('#2472ac'))
        p.setBrush(gradient)
        p.setPen(QPen(QColor('#f0b95a') if self.side in self.controller.anchors else QColor('#7193b5'), 2))
        p.drawEllipse(point, 17, 17)
        a, b, depth = self.PLANES[self.plane]
        orientation = self.controller.hands[self.side][:3, :3]
        for index, color in enumerate(['#ff7373', '#78d6a0', '#83aaff']):
            p.setPen(QPen(QColor(color), 2))
            vector = orientation[:, index]
            end = point + QPointF(38*vector[a], -38*vector[b])
            p.drawLine(point, end)
            p.drawText(end+QPointF(3, -3), 'XYZ'[index])
        p.setPen(QColor('#bdd0e6'))
        p.drawText(12, 21, f'{self.plane} 平面 · 40 mm / 格 · 滚轮调整 {"XYZ"[depth]}')
        p.drawText(12, self.height()-12, '独立手柄球 · 末端反馈不会改变此位置')
        p.end()


class FollowPanel(QWidget):
    def __init__(self, simulation, adapter, runtime, inputs, report, parent=None):
        super().__init__(parent)
        self.sim, self.adapter, self.runtime = simulation, adapter, runtime
        self.inputs, self.report = inputs, report
        self.on_prepared = None
        self.metadata = build_info(simulation)
        self.controller = FollowInput(inputs, lambda s: simulation.tcp_poses()[s],
                                      period=next(iter(simulation.arms.values())).config.period)
        self.manual_inputs = ManualFollowInput(self)
        self.loaded_trace = None
        self.hold_metrics = {}
        self.pads, self.observations, self.gripper_layouts = {}, {}, {}
        self.setObjectName('follow_page')
        root = QVBoxLayout(self)
        hint = QLabel('独立手柄球 → 相对位移 / 旋转 → TCP 目标。抓取时重新锚定；松手保留目标，仿真继续。')
        hint.setWordWrap(True)
        root.addWidget(hint)
        row = QHBoxLayout(); root.addLayout(row, 1)
        for side, title in [('left', '左臂'), ('right', '右臂')]:
            group = QGroupBox(title + ' · 虚拟手柄')
            layout = QVBoxLayout(group); row.addWidget(group)
            controls = QHBoxLayout(); layout.addLayout(controls)
            plane, axis = QComboBox(), QComboBox()
            plane.setObjectName(side + '_follow_plane')
            axis.setObjectName(side + '_follow_rotation_axis')
            plane.addItems(list(HandPad.PLANES)); axis.addItems(['绕 X', '绕 Y', '绕 Z']); axis.setCurrentIndex(1)
            controls.addWidget(QLabel('平移')); controls.addWidget(plane)
            controls.addWidget(QLabel('旋转')); controls.addWidget(axis)
            centre = QPushButton('视图居中'); controls.addWidget(centre)
            pad = HandPad(self.controller, side, self.error)
            self.pads[side] = pad; layout.addWidget(pad, 1)
            plane.currentTextChanged.connect(lambda value, p=pad: (setattr(p, 'plane', value), p.update()))
            axis.currentIndexChanged.connect(lambda value, p=pad: (setattr(p, 'rotation_axis', value), p._rates()))
            centre.clicked.connect(pad.recenter)
            instructions = QLabel('左键拖动位置 · 右键旋转 · 滚轮深度\n按住 Q/E 持续旋转，可与左键拖动同时进行')
            instructions.setWordWrap(True); layout.addWidget(instructions)
            label = QLabel(); label.setWordWrap(True)
            label.setMinimumHeight(label.fontMetrics().lineSpacing() * 5 + 8)
            label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            self.observations[side] = label; layout.addWidget(label)
            self.gripper_layouts[side] = layout
        bar = QHBoxLayout(); root.addLayout(bar)
        self.record_button = QPushButton('从 home 新录制')
        self.stop_button = QPushButton('结束录制 / 回放')
        self.save_button = QPushButton('保存输入与反馈')
        self.load_button = QPushButton('载入回放')
        self.replay_button = QPushButton('恢复录制起点并回放')
        self.repro_button = QPushButton('离线场景：下压＋旋转')
        for button in [self.record_button, self.stop_button, self.save_button, self.load_button, self.replay_button]:
            bar.addWidget(button)
        scenario = QHBoxLayout(); root.addLayout(scenario)
        self.repro_side, self.repro_angle = QComboBox(), QComboBox()
        self.repro_side.addItem('左臂', 'left'); self.repro_side.addItem('右臂', 'right')
        self.repro_angle.addItem('旋转 0.08 rad（4.6°）', .08)
        self.repro_angle.addItem('旋转 0.20 rad（11.5°）', .2)
        scenario.addWidget(self.repro_side); scenario.addWidget(self.repro_angle)
        scenario.addWidget(self.repro_button)
        scenario.addWidget(QLabel('仅离线：显式重置起点 → 下压＋旋转 → 保持 → 撤回'), 1)
        self.record_button.setToolTip('显式恢复仿真 home 和夹爪初始开度，清除旧目标，开始可重复的离线录制')
        self.record_button.setToolTip(self.record_button.toolTip() + '；替换当前缓存前请先保存')
        self.repro_button.setToolTip('仅仿真：恢复指定起始关节，左臂下压 12 cm＋旋转 0.08 rad，保持后撤回；不会连接真机')
        self.record_button.clicked.connect(lambda: self.run(self.start_recording))
        self.stop_button.clicked.connect(self.stop)
        self.save_button.clicked.connect(self.save)
        self.load_button.clicked.connect(self.load)
        self.replay_button.clicked.connect(lambda: self.run(lambda: self.play(self.loaded_trace)))
        self.repro_button.clicked.connect(lambda: self.run(self.reproduce))
        self.status = QLabel(self.controller.message); self.status.setWordWrap(True); root.addWidget(self.status)
        info = QLabel(f'运行 SDK {self.metadata["sdk_version"]} · native SHA256 {self.metadata["native_sha256"][:16]}\n'
                      f'{self.metadata["native_path"]}')
        info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.setWordWrap(True); root.addWidget(info)
        self.refresh({})

    def error(self, message):
        self.controller.message = '输入被拒绝：' + str(message)
        self.status.setText(self.controller.message)
        self.report(self.controller.message)

    def run(self, function):
        try:
            function()
        except Exception as exc:
            self.error(exc)
        self.refresh({})

    def set_active(self, active):
        for pad in self.pads.values(): pad.release_input()
        self.controller.set_enabled(active)
        self.refresh({})

    def prepare(self, initial_q):
        if not self.inputs.active:
            raise ValueError('Follow 没有输入权')
        q = np.asarray(initial_q, dtype=float)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError('Invalid initial joints')
        if np.any(q < self.sim.joint_limits[:, 0]) or np.any(q > self.sim.joint_limits[:, 1]):
            raise ValueError('Initial joints outside SDK limits')
        for i, arm in enumerate(self.sim.arms.values()):
            if not arm.planner.configuration_valid(q[i*6:(i+1)*6]):
                raise ValueError('Initial joints leave the startup configuration')
        # Recordings start with home fingers; validate before changing any state.
        if self.sim.collision and not self.sim.collision.check(
                self.adapter.grippers.full_q(positions=self.adapter.grippers.home, arm_q=q)).safe:
            raise ValueError('Recording start is blocked by collision geometry')
        self.runtime.reset_to_home()
        self.adapter.reset(q)
        for arm in self.sim.arms.values(): arm.time = 0.
        self.controller.reset()
        for pad in self.pads.values(): pad.release_input()
        if self.on_prepared:
            self.on_prepared()

    def start_recording(self):
        self.prepare(self.sim.home_q)
        self.controller.start_recording(self.metadata, self.sim.q)

    def play(self, trace):
        if trace is None:
            raise ValueError('先载入录制，或选择离线复现场景')
        trace.validate()
        if trace.metadata.get('environment_sha256') != self.metadata['environment_sha256']:
            raise ValueError('站点、碰撞配置或运动参数与录制不同；不能做等条件回放')
        if trace.period != self.controller.period:
            raise ValueError('Native control period differs')
        for entry in trace.commands:
            self.adapter.capabilities().validate(command_from_data(entry['command']))
        self.prepare(trace.initial_q)
        self.controller.start_playback(trace, metadata=self.metadata)

    def reproduce(self):
        self.loaded_trace = repro_trace(self.sim, self.metadata, side=self.repro_side.currentData(),
                                        angle=self.repro_angle.currentData())
        self.play(self.loaded_trace)

    def stop(self):
        self.controller.stop_recording()
        self.controller.stop_playback()
        self.controller.message = '已结束录制 / 回放；保留 SDK 最后目标'
        self.refresh({})

    def save(self):
        self.stop()
        if self.controller.trace is None:
            return self.error('没有可保存的录制')
        path, _ = QFileDialog.getSaveFileName(self, '保存 Follow 输入与逐周期反馈', 'follow.json', 'Follow JSON (*.json)')
        if path:
            self.run(lambda: self.controller.trace.save(path))

    def load(self):
        path, _ = QFileDialog.getOpenFileName(self, '载入 Follow 输入', '', 'Follow JSON (*.json)')
        if path:
            def read():
                self.loaded_trace = FollowTrace.load(path)
                changed = self.loaded_trace.metadata.get('native_sha256') != self.metadata['native_sha256']
                self.controller.message = '已载入；' + ('SDK 构建不同，可用于 A/B 对照' if changed else 'SDK 构建相同')
            self.run(read)

    def refresh(self, samples):
        c = self.controller
        for side, pad in self.pads.items():
            pad.setEnabled(c.active and c.playback is None)
            pad.update()
            actual = self.sim.tcp_poses()[side][:3, 3]
            target = self.sim.targets()[side][:3, 3]
            sample = samples.get(side, {})
            state = ('异常暂停' if c.paused else '碰撞限制（参考保持）' if sample.get('collision_hold')
                     else '运动受限' if sample.get('collision_limited') else '跟踪 / 待输入')
            hold = self.hold_metrics.get(side)
            held_text = '' if hold is None else f'\n目标固定 {hold["age"]:.2f} s · 稳定 0.5 s 后最大位移 {hold["drift"]*1000:.2f} mm'
            self.observations[side].setText(
                f'手柄位置 [m] {np.array2string(c.hands[side][:3,3], precision=3)}\n'
                f'目标 TCP {np.array2string(target, precision=3)}\n'
                f'仿真 TCP {np.array2string(actual, precision=3)} · 误差 {np.linalg.norm(target-actual)*1000:.2f} mm · {state}' + held_text)
        suffix = f' · 已记录 {c.step_index*c.period:.2f} s' if c.recording else ''
        self.status.setText((c.message if self.inputs.active else f'输入未启用；当前来源由其他页面或程序持有') + suffix)
        self.record_button.setEnabled(self.inputs.active and c.playback is None)
        self.replay_button.setEnabled(self.inputs.active and self.loaded_trace is not None)
        self.repro_button.setEnabled(self.inputs.active)

    def before_step(self):
        self.controller.before_step()

    def after_step(self, samples):
        capture = self.controller.recording or self.controller.play_capture
        self.controller.after_step(feedback(self.sim, samples) if capture else None)
        if self.controller.enabled:
            now = self.controller.step_index * self.controller.period
            for side, target in self.sim.targets().items():
                previous = self.hold_metrics.get(side)
                actual = self.sim.tcp_poses()[side][:3, 3]
                if previous is None or not np.allclose(previous['target'], target, atol=1e-10, rtol=0):
                    previous = dict(target=target.copy(), since=now, anchor=None, drift=0., age=0.)
                    self.hold_metrics[side] = previous
                previous['age'] = now - previous['since']
                if previous['age'] >= .5:
                    if previous['anchor'] is None:
                        previous['anchor'] = actual.copy()
                    previous['drift'] = max(previous['drift'], float(np.linalg.norm(actual-previous['anchor'])))

    def fault(self, error):
        self.controller.fault(error)
        self.refresh({})

    def reset(self):
        self.controller.reset()
        self.hold_metrics.clear()
        for pad in self.pads.values(): pad.release_input()
        self.refresh({})
