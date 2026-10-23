"""Optional metadata-driven target editor. No SDK, IK, timer or driver loop."""
import math

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget,
)

from .control import InputSession, JointTarget, TcpTarget
from .urdf import xyz_rpy


def _pose_values(pose):
    rotation = pose[:3, :3]
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return (*pose[:3, 3], roll, pitch, yaw)


class ControlPanel(QWidget):
    """Only explicit submit buttons produce commands through an InputSession.

    refresh() reads observations into separate labels, never target editors.
    sync_targets() explicitly copies current observations into a draft, without
    submitting. The host chooses its own observation refresh schedule.
    """
    def __init__(self, inputs: InputSession, *, xyz_range=0.3, parent=None):
        super().__init__(parent)
        if not np.isfinite(xyz_range) or xyz_range <= 0:
            raise ValueError('xyz_range must be finite and positive')
        self.inputs = inputs
        self.session = inputs.controller
        self.xyz_range = float(xyz_range)
        self.joint_editors = {}
        self.tcp_editors = {}
        self.observed_joints = {}
        self.observed_tcps = {}
        self.joint_buttons = {}
        self.tcp_buttons = {}
        self.last_receipt = None
        layout = QVBoxLayout(self)
        self.source_status = QLabel()
        layout.addWidget(self.source_status)
        self.status = QLabel('Edit a target, then submit. Submission does not mean sent or achieved.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        capabilities = self.session.capabilities
        if capabilities.joint_targets and capabilities.groups:
            page, groups = QWidget(), QVBoxLayout()
            page.setLayout(groups)
            for group, names in capabilities.groups.items():
                box = QGroupBox(group)
                form = QFormLayout(box)
                editors, observed = {}, {}
                for joint in names:
                    spec = capabilities.joints[joint]
                    edit = QDoubleSpinBox()
                    edit.setDecimals(9)
                    bounds = [spec.to_display(spec.lower) if spec.lower is not None else (-1e12 if spec.display_scale > 0 else 1e12),
                              spec.to_display(spec.upper) if spec.upper is not None else (1e12 if spec.display_scale > 0 else -1e12)]
                    edit.setRange(*sorted(bounds))
                    edit.setSingleStep(0.1)
                    edit.setSuffix(' ' + spec.display_unit)
                    edit.setObjectName('joint:' + group + ':' + joint)
                    form.addRow(joint + ' target', edit)
                    label = QLabel()
                    form.addRow(joint + ' observed', label)
                    editors[joint], observed[joint] = edit, label
                self.joint_editors[group], self.observed_joints[group] = editors, observed
                button = QPushButton('Submit joint target · ' + group)
                button.clicked.connect(lambda checked=False, selected=group: self.submit_joints(selected))
                form.addRow(button)
                self.joint_buttons[group] = button
                groups.addWidget(box)
            self._add_page(page, 'Joint targets')
        if capabilities.tcp_targets:
            page, groups = QWidget(), QVBoxLayout()
            page.setLayout(groups)
            for tcp, spec in capabilities.tcp_targets.items():
                box = QGroupBox(f'{tcp} · {spec.group} · world frame')
                form, editors = QFormLayout(box), {}
                for axis in ('x', 'y', 'z', 'roll', 'pitch', 'yaw'):
                    edit = QDoubleSpinBox()
                    edit.setDecimals(9)
                    edit.setRange(-1e12, 1e12) if axis in 'xyz' else edit.setRange(-math.pi, math.pi)
                    edit.setSingleStep(0.001 if axis in 'xyz' else 0.01)
                    edit.setSuffix(' m' if axis in 'xyz' else ' rad')
                    edit.setObjectName('tcp:' + tcp + ':' + axis)
                    form.addRow(axis + ' target', edit)
                    editors[axis] = edit
                self.tcp_editors[tcp] = editors
                observed = QLabel()
                observed.setWordWrap(True)
                form.addRow('Observed TCP', observed)
                self.observed_tcps[tcp] = observed
                button = QPushButton('Submit TCP target · ' + tcp)
                button.clicked.connect(lambda checked=False, selected=tcp: self.submit_tcp(selected))
                form.addRow(button)
                self.tcp_buttons[tcp] = button
                groups.addWidget(box)
            self._add_page(page, 'Cartesian targets')
        self.sync_button = QPushButton('Copy observed state to target drafts')
        self.sync_button.clicked.connect(lambda: self.sync_targets())
        layout.addWidget(self.sync_button)
        self.sync_targets()
        self.refresh()

    def _add_page(self, page, label):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, label)

    def _submit(self, command):
        try:
            receipt = self.inputs.submit(command)
        except Exception as exc:
            self.status.setText(f'Target rejected: {exc}')
            return None
        self.last_receipt = receipt
        self.status.setText(f'Target submitted #{receipt.sequence} by {receipt.source}; sent/achieved unknown.')
        return receipt

    def submit_joints(self, group):
        values = {joint: self.session.capabilities.joints[joint].from_display(edit.value())
                  for joint, edit in self.joint_editors[group].items()}
        return self._submit(JointTarget(group, values))

    def submit_tcp(self, tcp):
        values = [edit.value() for edit in self.tcp_editors[tcp].values()]
        return self._submit(TcpTarget(tcp, xyz_rpy(values[:3], values[3:])))

    def sync_targets(self):
        state = self.session.snapshot()
        for editors in self.joint_editors.values():
            for joint, edit in editors.items():
                with QSignalBlocker(edit):
                    edit.setValue(self.session.capabilities.joints[joint].to_display(state.joint_positions[joint]))
        for tcp, editors in self.tcp_editors.items():
            pose = state.tcp_poses.get(tcp)
            # No guessed identity target for a TCP with no observation.
            for edit in editors.values():
                edit.setEnabled(pose is not None)
            if pose is not None:
                for axis, value in zip(editors, _pose_values(pose), strict=True):
                    edit = editors[axis]
                    with QSignalBlocker(edit):
                        if axis in 'xyz':
                            edit.setRange(value - self.xyz_range, value + self.xyz_range)
                        edit.setValue(value)

    def refresh(self):
        state = self.session.snapshot()
        active = self.inputs.active
        self.source_status.setText(f'Active input: {self.session.active_source or "none"}; this panel: {self.inputs.source}')
        for group, observed in self.observed_joints.items():
            self.joint_buttons[group].setEnabled(active)
            for joint, label in observed.items():
                spec = self.session.capabilities.joints[joint]
                label.setText(f'{spec.to_display(state.joint_positions[joint]):.6f} {spec.display_unit}')
        for tcp, label in self.observed_tcps.items():
            pose = state.tcp_poses.get(tcp)
            label.setText('Unavailable' if pose is None else 'world XYZ [m] / RPY [rad]: ' +
                          ', '.join(f'{v:.6f}' for v in _pose_values(pose)))
            self.tcp_buttons[tcp].setEnabled(active and all(e.isEnabled() for e in self.tcp_editors[tcp].values()))
        return state
