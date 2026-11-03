"""Offline finger motion using the station's opening calibration and URDF limits."""
import json
from time import monotonic
import xml.etree.ElementTree as ET

import numpy as np

from ..control import JointControl, JointTarget


class EliteGrippers:
    """Rate-limited simulated joints, not motor feedback or a hardware driver.

    The native station guard receives the same expanded finger configuration
    as the renderer. Arm planning/IK still belongs entirely to libpyelite.
    """
    def __init__(self, simulation):
        self.simulation = simulation
        elements = {j.attrib['name']: j for j in ET.parse(simulation.urdf_path).getroot().findall('joint')}
        profile = json.loads((simulation.assembly.root / 'simulation_profile.json').read_text())
        calibration = simulation.settings.get('native_collision_gripper_calibration',
                                               profile['native_collision_gripper_calibration'])
        self.drivers, self.knots, self.specs, self.velocities = {}, {}, {}, {}
        for side in simulation.arms:
            drivers = calibration[side]
            if len(drivers) != 1:
                raise ValueError('Elite opening control requires one calibrated driver joint per gripper')
            joint, values = next(iter(drivers.items()))
            element = elements[joint]
            knots = np.asarray(values, dtype=float)
            limit = element.find('limit')
            if (element.find('mimic') is not None or element.attrib['type'] != 'revolute'
                    or knots.ndim != 2 or knots.shape[1] != 2 or len(knots) < 2
                    or not np.isfinite(knots).all() or np.any(np.diff(knots[:, 0]) <= 0)
                    or knots[0, 0] != 0 or knots[-1, 0] != 1
                    or np.any(np.diff(knots[:, 1]) <= 0)
                    or knots[:, 1].min() < float(limit.get('lower'))
                    or knots[:, 1].max() > float(limit.get('upper'))):
                raise ValueError(f'Invalid Elite gripper opening calibration: {joint}')
            self.drivers[side], self.knots[side] = joint, knots.copy()
            self.specs[joint] = JointControl('rad', knots[0, 1], knots[-1, 1], 'deg', 180 / np.pi)
            self.velocities[joint] = float(limit.get('velocity'))
            if not np.isfinite(self.velocities[joint]) or self.velocities[joint] <= 0:
                raise ValueError(f'Invalid finger velocity limit: {joint}')
        defaults = simulation.assembly.viewer_joint_defaults()
        self.home = {n: defaults[n] for n in self.drivers.values()}
        self.positions, self.targets = self.home.copy(), self.home.copy()
        self.mimics = {n: (e.find('mimic').get('joint'), float(e.find('mimic').get('multiplier', '1')),
                           float(e.find('mimic').get('offset', '0')))
                       for n, e in elements.items() if e.find('mimic') is not None}
        self._guard_configured = False
        self._sequence = 0
        self.blocked = False

    def target(self, side, opening):
        opening = float(opening)
        if not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError('Gripper opening must be finite in [0, 1] (closed to open)')
        knots = self.knots[side]
        return JointTarget(side + '_gripper', {self.drivers[side]: float(np.interp(opening, knots[:, 0], knots[:, 1]))})

    def opening(self, side):
        knots = self.knots[side]
        return float(np.interp(self.positions[self.drivers[side]], knots[:, 1], knots[:, 0]))

    def expanded(self, positions=None):
        values = dict(self.positions if positions is None else positions)
        pending = dict(self.mimics)
        while pending:
            resolved = [n for n, (parent, _, _) in pending.items() if parent in values]
            if not resolved:
                raise ValueError('Unresolved gripper mimic joints')
            for n in resolved:
                parent, scale, offset = pending.pop(n)
                values[n] = values[parent] * scale + offset
        return values

    def full_q(self, positions=None, arm_q=None):
        sim = self.simulation
        q = sim.viewer_q(arm_q)
        for name, value in self.expanded(positions).items():
            q[sim.model.joints[sim.model.getJointId(name)].idx_q] = value
        return q

    def _publish(self):
        sim = self.simulation
        if sim.guard is None:
            return
        values = self.expanded()
        names = sorted(values)
        if not self._guard_configured:
            sim.guard.configure_grippers(names, [0 if n.startswith('left_') else 1 for n in names], .1)
            self._guard_configured = True
        self._sequence += 1
        # These are fresh offline simulated samples on the native monotonic
        # freshness clock, independent of the recording's simulation timeline.
        sim.guard.update_grippers([values[n] for n in names], [monotonic()] * 2,
                                  [self._sequence] * 2, [True, True])

    def advance(self):
        sim = self.simulation
        period = next(iter(sim.arms.values())).config.period
        candidate = {n: q + float(np.clip(self.targets[n] - q, -self.velocities[n] * period,
                                         self.velocities[n] * period)) for n, q in self.positions.items()}
        self.blocked = False
        if candidate != self.positions:
            if sim.collision is not None:
                result = sim.collision.check_segment(self.full_q(), self.full_q(candidate))
                self.blocked = not result.safe
            if not self.blocked:
                self.positions = candidate
            self._publish()
        elif self._guard_configured:
            self._publish()

    def reset(self, *, home=False):
        if home:
            self.positions = self.home.copy()
        self.targets = self.positions.copy()
        self.blocked = False
        if self._guard_configured:
            self._publish()


def make_gripper_panel(inputs, grippers, joint_layouts, *, ik_layouts, on_submit=None):
    """Mount synchronized opening controls below each arm's Joint and IK inputs."""
    from types import SimpleNamespace
    from PySide6.QtCore import Qt, QSignalBlocker
    from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
    box = SimpleNamespace(sliders={}, editors={}, observed={}, rows={}, statuses={})
    box.ik = SimpleNamespace(sliders={}, editors={}, observed={}, rows={}, statuses={})
    panels = (box, box.ik)

    def set_target(side, value):
        for panel in panels:
            with QSignalBlocker(panel.sliders[side]), QSignalBlocker(panel.editors[side]):
                panel.sliders[side].setValue(round(value * 10))
                panel.editors[side].setValue(value)

    def submit(side, value):
        set_target(side, value)
        try:
            receipt = inputs.submit(grippers.target(side, value / 100))
            message = f'目标 {value:.1f}% 已提交 #{receipt.sequence}'
        except Exception as exc:
            for panel in panels:
                panel.statuses[side].setText(f'夹爪目标被拒绝: {exc}')
            return
        for panel in panels:
            panel.statuses[side].setText(message)
        if on_submit is not None:
            on_submit()

    def add_row(panel, side, parent_layout, *, prefix=''):
        row = QWidget()
        row.setObjectName(prefix + side + '_gripper_row')
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 1, 0, 1)
        controls, feedback = QHBoxLayout(), QHBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(feedback)
        label = QLabel('夹爪开度')
        label.setMinimumWidth(92)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        slider.setTracking(True)
        slider.setObjectName(prefix + side + '_gripper_slider')
        slider.setToolTip('0% 闭合，100% 张开；拖动即提交目标')
        edit = QDoubleSpinBox()
        edit.setRange(0, 100)
        edit.setDecimals(1)
        edit.setSuffix(' %')
        edit.setMinimumWidth(112)
        observed = QLabel()
        status = QLabel('0% 闭合 / 100% 张开')
        status.setWordWrap(True)
        panel.sliders[side], panel.editors[side], panel.observed[side] = slider, edit, observed
        panel.rows[side], panel.statuses[side] = row, status
        controls.addWidget(label)
        controls.addWidget(slider, 1)
        controls.addWidget(edit)
        feedback.addWidget(observed)
        feedback.addWidget(status, 1)
        parent_layout.addWidget(row)
        slider.valueChanged.connect(lambda value: submit(side, value / 10))
        edit.valueChanged.connect(lambda value: submit(side, value))

    for side in grippers.drivers:
        add_row(box, side, joint_layouts[side])
        add_row(box.ik, side, ik_layouts[side], prefix='ik_')

    def refresh():
        for panel in panels:
            for side in grippers.drivers:
                panel.sliders[side].setEnabled(inputs.active)
                panel.editors[side].setEnabled(inputs.active)
                panel.observed[side].setText(f'实际 {100 * grippers.opening(side):.1f}%'
                                           + (' · 碰撞约束保持' if grippers.blocked else ''))

    def sync():
        for side in grippers.drivers:
            set_target(side, grippers.opening(side) * 100)
        refresh()
    box.refresh, box.sync = refresh, sync
    sync()
    return box
