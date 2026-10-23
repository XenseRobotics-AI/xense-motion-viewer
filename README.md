# Robot Motion Workbench · 机器人运动工作台

独立的 Python 库，为机械臂提供公共控制接口、状态观测、完整 policy 未来轨迹和基于 URDF 的 Rerun 可视化。Qt 与 policy 都是输入来源；机器人自己的 SDK 负责 IK、运动规划、碰撞检查、控制频率和设备生命周期，工作台通过薄适配器接入这些能力。

## 仓库与源码位置

本机仓库根目录：`/home/olorin/Olorin/robot-motion-workbench`

- 本文档：`/home/olorin/Olorin/robot-motion-workbench/README.md`
- Python 库源码：`/home/olorin/Olorin/robot-motion-workbench/src/robot_motion_workbench/`
- 包名：`robot-motion-workbench`；Python 导入名：`robot_motion_workbench`
- 当前版本：`0.4.0`；要求 Python 3.12 或更新版本。

这是独立于 LeRobot/Elite 工作区的本地 Git 仓库。HTML 工程报告另存于 `/home/olorin/Olorin/03_ReportDoc/`；报告目录不包含库源码。将仓库复制到其他位置后，以下从仓库根目录执行的相对路径命令仍然适用。

```text
robot-motion-workbench/
├── README.md                       # 本中文说明
├── pyproject.toml                  # 包元数据、依赖和命令行入口
├── src/robot_motion_workbench/      # Python 库源码
│   ├── contracts.py                # 模型与状态契约
│   ├── control.py                  # 能力、目标、输入会话与公共控制接口
│   ├── urdf.py                     # URDF 解析与显示 FK
│   ├── qt.py                      # 根据能力生成的 Qt 输入面板
│   ├── rerun_renderer.py           # 模型与状态显示
│   ├── policy.py                  # 完整动作块、步骤选择与提交
│   ├── streams.py                 # 有界异步显示队列
│   ├── policy_renderer.py          # 预测、提交和反馈可视化
│   ├── adapters/                  # Elite、Flexiv 与参考运动适配器
│   ├── examples/                  # 通用 URDF 与外部 SDK 回调示例
│   └── assets/                    # 随包分发的原创七轴演示模型
├── docs/                          # 适配器与 policy 详细契约
├── tests/                         # 产品测试
├── scripts/                       # 验证辅助脚本
└── evidence/                      # 实施记录、安装包、录制和验证证据
```

## 功能与定位

- **接入方式：URDF + 公共接口适配。** URDF 描述模型，适配器负责 SDK 的状态读取、目标提交、关节名称与顺序映射。控制能力由具体机器人声明。
- **输入与显示分离。** Qt 面板和 policy 使用同一个 `RobotSession`；Rerun 负责模型、轨迹、观测数据的显示与录制。
- **完整未来轨迹。** 一次接收整个动作块，保留未来时间偏移；选中步骤的提交与实际规划、发送、反馈分别记录。
- **依赖按需安装。** 核心仅依赖 NumPy；Qt、Rerun 和厂商 SDK 都是可选依赖。
- **保留现有 Elite 能力。** 复用原生规划器、碰撞保护和原有 Qt 操作，也提供通用 Qt 面板。

包内带有原创的基础几何七轴模型。另已使用外部 Flexiv Rizon4 URDF 验证全部七轴、第七轴法兰旋转以及非单位基座／工具变换。该演示使用显式的数值跟随状态，不属于动力学仿真或实机验证。

## 安装与快速运行

### 从当前仓库安装

先进入仓库根目录；如果已有 `.venv`，可以直接使用：

```sh
cd /home/olorin/Olorin/robot-motion-workbench
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/rmw-demo --steps 100
.venv/bin/rmw-callback-demo
```

`rmw-demo` 接收手工构造的 12 步关节参考，显式提交选中的步骤，再推进受速率限制的数值跟随器。JSON 输出包含每个轴的影响、初末状态、基座／工具变换、提交回执，以及不可用的发送／规划遥测。

`rmw-callback-demo` 演示 SDK 向量顺序与 URDF 顺序不同的映射，并验证“接受目标”不会直接改写反馈。这两个命令只需要核心依赖与随包 URDF；安装后可从任意工作目录运行，不依赖本机私有模型路径。

### 安装显示依赖并查看录制

在仓库根目录执行：

```sh
.venv/bin/python -m pip install -e '.[rerun,qt]'
.venv/bin/rmw-demo --save portable-policy.rrd
.venv/bin/python -m rerun rrd verify portable-policy.rrd
.venv/bin/rmw-policy-replay portable-policy.rrd

.venv/bin/rmw-demo-qt --save portable-qt.rrd
# 关闭 Qt 面板，完成录制后再回放：
.venv/bin/rmw-policy-replay portable-qt.rrd
```

`rmw-demo-qt` 打开通用输入面板；`--save` 保存 RRD 文件，**不会自动启动或连接实时 Rerun Viewer**。关闭面板后回放保存的文件。Elite 的 `rmw-elite-qt` 保留独立的实时 Rerun Viewer，可用 `--no-viewer` 关闭。

已验证的显示依赖固定为 `rerun-sdk==0.38.1`、Trimesh 5.1.1、PyCollada 0.9.3；请使用匹配的 Viewer。这是经过验证的版本组合，不表示它们始终是最新版本。Qt 使用可选依赖 PySide6。

离屏 Qt 验证可显式修改目标草稿并点击一次提交：

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/rmw-demo-qt --no-show --ticks 30 \
  --exercise --save portable-qt.rrd
```

不使用 `--exercise` 时，编辑或刷新面板不会提交运动目标，只有提交按钮会提交。参考适配器只声明关节目标，因此不显示笛卡尔控制页；它可以观测声明的工具位姿，但不提供 IK。

### 使用已验收的 wheel

本机安装包位于 `evidence/phase4/wheels/robot_motion_workbench-0.4.0-py3-none-any.whl`。也可以将它复制到其他环境安装，无需复制整个仓库。以下仍从仓库根目录执行：

```sh
.venv/bin/python -m pip install evidence/phase4/wheels/robot_motion_workbench-0.4.0-py3-none-any.whl
# 如需 Qt 和 Rerun：
.venv/bin/python -m pip install 'evidence/phase4/wheels/robot_motion_workbench-0.4.0-py3-none-any.whl[rerun,qt]'
```

## 使用外部 Rizon4 或其他 URDF

显式传入 URDF，并保留相对 mesh 目录结构。以下以已验证的 `xense-description` Rizon4 资源为例：

```sh
.venv/bin/rmw-demo --urdf /path/to/rizon4/urdf/rizon4.urdf \
  --tool-link comp_Rizon4_Rizon4-062123_flange --instance-id rizon4 \
  --steps 100 --save rizon4-policy.rrd
QT_QPA_PLATFORM=offscreen .venv/bin/rmw-demo-qt \
  --urdf /path/to/rizon4/urdf/rizon4.urdf \
  --tool-link comp_Rizon4_Rizon4-062123_flange --no-show --ticks 30 \
  --exercise --save rizon4-qt.rrd
.venv/bin/python -m rerun rrd verify rizon4-policy.rrd rizon4-qt.rrd
```

将示例路径替换为你有权使用的模型位置。mesh 路径相对于 URDF 解析；对于 `package://name/path` 引用，按需添加 `--package-root name=/path/to/package`。

- `--base X Y Z R P YAW`：URDF 根坐标系到世界坐标系的变换，即 `world_from_root`。
- `--tool X Y Z R P YAW`：工具坐标系到其父 link 的变换，即 `parent_link_from_tool`。
- 平移单位为米，旋转单位为弧度。演示默认使用包含平移和旋转的基座 `[.3,-.2,.12,.1,-.15,.35]` 与工具 `[.04,-.02,.12,.2,-.1,.3]` 变换。
- `--tool-link` 指定工具父 link；未指定时，示例选择已验证树结构中的最后一个后代 link。
- 独立关节名称取自 URDF 顺序；实际 SDK 集成必须显式提供 SDK 向量顺序映射。

模型、参考适配器、控制会话、Qt、policy 与显示器均不包含针对六轴／七轴的分支。**仅换 URDF 可以运行数值参考演示；要控制另一台机器人，还需要实现它的公共接口适配。**

外部 Rizon4 资源源自 Flexiv 官方 `flexiv_description` 的提交 `8b8452105c5faf088f456db18b0e2581b16fc630`，许可证为 Apache-2.0。资源旁的 `README.md`、`asset.json` 和 `LICENSES/LICENSE` 说明了上游输入、兼容命名、惯性结构处理与相对 mesh 路径。本库 wheel 不包含 Flexiv 模型或厂商二进制。

随包基础几何模型为原创示意模型，按 CC0-1.0 提供，见 [模型许可](src/robot_motion_workbench/assets/demo_license.txt)。它不是 Rizon4 模型，也不是标定后的机器人模型。

<a id="connect-an-external-sdk"></a>

## 接入外部机械臂控制库

[可运行的回调示例](src/robot_motion_workbench/examples/callback_sdk.py) 提供 `make_callback_session`。传入 `RobotDescription`、已核对的 `JointControl` 元数据、显式 `sdk_joint_names`、状态向量读取函数和目标向量写入函数，即可通过 `CallbackRobotAdapter` 得到 Qt 与 policy 共用的 `RobotSession`。

以下代码假设调用方已准备好描述、能力元数据和 SDK 对象；示例中的假 SDK 可以替换成已配置好的外部对象：

```python
from robot_motion_workbench import JointTarget
from robot_motion_workbench.examples.callback_sdk import make_callback_session

session = make_callback_session(
    description, joint_controls, sdk_joint_names=sdk_order,
    read_positions=sdk.read_positions, write_positions=sdk.accept_positions,
)
producer = session.activate('application')
state = session.snapshot()
receipt = producer.submit(JointTarget('arm', dict(state.joint_positions)))
assert receipt.status == 'submitted'
producer.close()
```

模型描述包含有序 URDF 关节名称、资源根目录、`world_from_root` 安装变换与命名工具帧。关节单位为弧度／米；控制分组和能力来自机器人适配器，声明的静态界限用于校验输入，不能替代 SDK 的动态限制和安全检查。

变换统一为作用于列向量的刚体 4×4 `parent_from_child` 矩阵；TCP 观测与目标使用世界坐标系。命令、描述和快照均复制输入数据，并保持不可变。

### Flexiv 七轴适配器

可选的 [FlexivJointAdapter](src/robot_motion_workbench/adapters/flexiv.py) 接收调用方持有、已经配置好的 `flexiv_rt.JointImpedanceControl` 对象，仅映射 `is_running()`、`get_state().q` 和 `set_target_joints(positions)`。

构造适配器时不会调用上述方法，也不会创建 Robot、连接设备、使能、启停控制或切换模式。必须提供**按 SDK J1～J7 顺序排列的七个 URDF 关节名称**，以及调用方核对过的弧度制控制元数据；不会根据名称后缀或模型顺序猜测映射。

```python
from robot_motion_workbench import RobotSession
from robot_motion_workbench.adapters.flexiv import FlexivJointAdapter

adapter = FlexivJointAdapter(
    configured_joint_control, description, sdk_joint_names=sdk_j1_to_j7,
    joint_controls=verified_joint_controls,
)
session = RobotSession(adapter)
```

已检查的实际 SDK setter 写入共享内存，返回仅表示 **submitted（已提交）**，不表示已发送、已被实时循环消费、已执行或已到达。

`get_state().q` 没有来源采样时间戳。该适配器的 `RobotState.time` 是相对于起点的**主机轮询完成时间**；使用 `SampleTime(state.time)` 时，`source_time` 与 `source_clock` 必须保持缺省。最近一次主机轮询不能证明共享内存中的样本也是最新的；实际发送数据和来源采样时间仍标为不可用。

SDK 的 TCP 数据未接入，因为工具配置与坐标标定需由应用确认；适配器也不声明笛卡尔控制。接口映射使用与已检查 C++ 绑定一致的替身对象验证，未连接 Flexiv 实机或运行其 SDK。

### 建议的依赖组织

应用分别依赖本工作台包与机器人控制库，并在应用或独立集成包中维护薄适配器。厂商 SDK 可以作为应用依赖，也可以由应用通过可选 git submodule 管理。

公共核心不内置厂商 SDK，也不反向导入应用；厂商 SDK 本身无需依赖工作台。基础安装不要求任何厂商子模块。

## 输入、policy、遥测与生命周期

`RobotSession.activate(source)` 返回唯一有效的 `InputSession`。调用 `submit(JointTarget(...))` 或 `submit(TcpTarget(...))` 时，先校验名称、单位、控制组、静态界限和刚体变换，再同步调用适配器。回执仅报告提交结果，反馈始终通过独立快照读取。

`activate(..., replace=True)` 显式替换当前输入源，使旧输入会话失效。Qt 使用 `ControlPanel(session.activate('qt'))`；policy 使用 `PolicyInput(session.activate('policy'), display_worker.publish)`，依次调用：

1. `accept(chunk)`：接收并显示完整动作块。
2. `select(index, now=...)`：选择步骤，只改变选择状态。
3. `execute_selected(now=...)`：通过与 Qt 相同的输入接口提交所选的一步。

输入源切换不会停止机器人，也不会取消已经接受的目标；关闭输入会话仅释放该输入源。动作执行时序和停止策略由机器人控制库与应用决定。

完整 `PolicyChunk` 接收反归一化后的绝对关节位置或世界坐标 TCP 位姿，并包含明确的时间偏移、来源／主机时钟元数据及到期时间。VLA 推理、相对动作解码与坐标转换由应用完成。

未来预测始终作为参考显示；当前模型几何与 TCP 历史来自可用反馈。参考示例通过显示 FK 生成完整未来关节骨架与工具路径，不将其当作未来测量。新的“不可用”反馈不会改写保留几何的来源；真实发送遥测与提交返回后产生的缺失占位分别记录。

显示几何只要求模型描述和完整的命名关节快照；提交命令还需要控制能力。TCP 观测、目标位姿、规划值、实际发送遥测、来源时间戳、提交关联和关节预览均为可选信息。未知信息应省略或明确标为不可用，不能由 setter 成功推造。详见 [适配器契约](docs/adapters.md) 和 [policy 语义](docs/policy.md)。

应用负责 SDK 连接、安全、启停、模式、实时线程、轮询频率、同步和时钟转换。会话调用同步且串行，不会自动把回调转交到 SDK 所属线程；Qt、policy 与 SDK 操作应在正确线程执行，或由应用调度到该线程。

参考 Qt 示例在 GUI 线程推进数值状态并刷新快照；policy 输入留在所属线程，只有复制后的数据进入有界 `VisualizationWorker`。队列溢出时可能丢弃显示事件，应检查计数并在退出时关闭／排空队列。渲染和回放不接收控制器，不会驱动硬件。

需要实时显示的应用，可以为自己持有的 `rerun.RecordingStream` 配置连接，再传入公共 `robot_motion_workbench.rerun_renderer.RerunRenderer`。数据流与 Viewer 生命周期、渲染线程及 SDK 线程调度仍由应用负责。

## 现有 Elite 集成

Elite 还需要保留的当前生产源码构建包：

- 路径：`evidence/wheels/libpyelite-0.1.0-cp312-cp312-linux_x86_64.whl`
- SHA-256：`dea977961be77d128cf47f8b077ccb6cd1cec88bada0e00a69f55eb8275b34d6`
- 源码与构建来源：[阶段一实施记录](evidence/phase1-implementation.json)。仅凭版本号无法区分较旧的规划器代码。

该原生 wheel 适用于 CPython 3.12／Linux x86-64；其他平台需要兼容的 SDK 安装。本次发布未重复进行原生编译。

从仓库根目录安装该 SDK wheel 和工作台的可选依赖后运行：

```sh
.venv/bin/python -m pip install evidence/wheels/libpyelite-0.1.0-cp312-cp312-linux_x86_64.whl
.venv/bin/python -m pip install -e '.[elite,rerun,qt]'
.venv/bin/rmw-elite-record --station diagonal-08 --collision-profile auto --steps 100 \
  --jog-m 0.005 --save elite.rrd
.venv/bin/rmw-elite-qt --station diagonal-08 --collision-profile auto
.venv/bin/rmw-elite-qt --ui generic --collision-profile auto
.venv/bin/rmw-elite-policy --kind joint --steps 120 --save elite-policy.rrd
```

原有关节／笛卡尔滑条、诊断信息、recipe 工具偏移、碰撞保护及原生 `StationSimulation.step` 继续负责相应行为。`--recipe path.yaml` 使用原有配置加载器，工位不匹配时拒绝运行。

碰撞配置保持 SDK 语义，Qt 碰撞显示开关只控制 Rerun 可见性。`--no-viewer --no-show` 支持离屏 Elite Qt 冒烟验证。Elite 离线示例中的原生规划值和反馈明确标为模拟数据；实际发送证据仍不可用。

## 命令行入口

| 入口 | 用途 | 可选依赖 |
|---|---|---|
| `rmw-demo` | 随包或外部 URDF 的参考运动，输出 JSON，可选 RRD | 使用 `--save` 时需要 `rerun` |
| `rmw-callback-demo` | 最小可运行 SDK 回调映射 | 无 |
| `rmw-demo-qt` | 根据能力生成的通用输入面板与参考运动 | `qt`；使用 `--save` 时还需要 `rerun` |
| `rmw-elite-record`、`rmw-elite-qt` | 原有 Elite 原生沙盒 | `elite,rerun`；Qt 入口还需要 `qt` |
| `rmw-elite-policy` | 完整未来参考与现有 Elite 原生规划器 | `elite,rerun` |
| `rmw-policy-replay` | 只读 RRD 查看与回放 | `rerun` |

## 开发与验证

项目采用 `src/` 布局和标准 setuptools。在仓库根目录执行：

```sh
.venv/bin/python -m pip install -e '.[rerun,qt,test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m pip wheel --no-deps . -w evidence/phase4/wheels
```

设置 `RMW_RIZON4_URDF` 可纳入外部授权模型相关测试；原生 Elite 测试还需要匹配的 SDK 依赖。完整验收环境中，80 项产品测试与 17 项独立测试通过；仅安装核心或缺少外部模型时，不代表具备完整验收环境。

发布包已分别安装到新的核心依赖环境和完整环境，并从源码目录外运行。验证覆盖：无厂商／Qt／Rerun 时的核心导入与演示、原生 Elite 回归、实际 Qt 信号与输入所有权、Rizon4 全七轴与 J7 姿态、来源／主机时间分离、RRD Arrow 数据读取及 Rerun 实际截图。

实施命令、日志、安装包／模型哈希与限制见 [阶段四实施记录](evidence/phase4-implementation.json)，较早阶段的证据也保留在 `evidence/` 中。独立审查与中文 HTML 报告保存在本机 `/home/olorin/Olorin/03_ReportDoc/`，不影响本库运行。

## 支持范围与已知限制

- 显示运动学支持连通树形 URDF：固定、转动、连续、移动关节，以及递归推导的 mimic 关节。
- 支持基础几何、STL／DAE／Trimesh 支持的网格、mesh 缩放及独立 visual origin。
- 不支持浮动／平面关节、闭环机构或运行中替换几何；材质使用 URDF 或默认 RGBA，不呈现纹理贴图。
- 通用 Qt 示例提供输入与录制，不自动启动实时 Viewer；Elite Qt 保留实时显示。
- 较长的关节名称在参考曲线图中可能重叠，曲线与原始数值仍完整。
- 本库不提供硬实时保证、动力学精度保证或实机验证结论。通用会话不负责连接、自动重试、停止命令或通用控制器。
