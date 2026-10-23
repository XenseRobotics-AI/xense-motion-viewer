# Robot Motion Workbench

Elite CS66 双臂的离线交互与策略参考展示。输入使用模拟目标，运动仍由已有
`libpyelite.sim.StationSimulation`、原生规划器与碰撞检查执行；不运行 VLA 推理，
也不连接或控制真机。

## 一次性选择环境

在本项目目录使用项目自己的环境，每个新终端先激活一次：

```sh
source .venv/bin/activate
```

当前环境已包含原生 Elite wheel、Qt 和固定版本的 Rerun。不要混用另一个 Conda
环境中的入口。更新本项目源码后的本地入口只需 `python -m pip install --no-deps -e .`。
全新环境的安装说明见下方“开发与依赖”。

## 日常使用：两个命令

```sh
rmw-elite-qt --collision-profile auto
rmw-elite-policy
```

第一个打开原有的关节与 XYZ/RPY 输入面板和独立 Rerun Viewer。拖动提交的是目标，
机器人按原生速度、加速度与碰撞约束运动；不是瞬移。碰撞包络开关只改变显示，
不会关闭碰撞检查。目标不可达或安全检查锁存时，按面板说明显式重置。

第二个默认打开 Viewer，按墙钟节奏展示约 12 秒的模拟策略执行。蓝色是整段未来
参考，橙色是当前选中的参考点，绿色是已收到的仿真 TCP 轨迹，机器人模型显示
仿真实际状态；它们在**同一个 3D 视图**内。没有单独的轨迹视窗。
`--telemetry` 可增加文本状态栏，`--kind joint` 可改用关节参考。
重复参考围绕启动姿态生成，不会逐段累加漂向远处；仍由原生规划器判断是否可达。

需要同时看和录制时，加 `--save`：

```sh
rmw-elite-qt --collision-profile auto --save elite-qt.rrd
rmw-elite-policy --save elite-policy.rrd
rmw-policy-replay elite-policy.rrd
```

应用持有带 64 MiB 缓冲上限的 Rerun 服务端，并同时写入只创建一次的文件 sink。
关闭 Viewer 不停止仿真或录制；点击“打开 Rerun”可重连并恢复静态模型、当前姿态和
最新碰撞显示。已连接时重复点击不重置相机。重连会追加完整场景快照，文件保留重开前后的显示样本，
关闭 Qt 或等策略示例结束后再回放。Qt 默认 Viewer 端口为 9876，策略示例为
9877；可用 `--port` 改成空闲端口。以 `--no-viewer` 启动时不开实时服务，
打开按钮禁用；录制请在结束后回放。

自动验证时可显式省略 Viewer 或加速有限步数：

```sh
QT_QPA_PLATFORM=offscreen rmw-elite-qt --no-viewer --no-show \
  --collision-profile auto --save elite-qt-smoke.rrd
rmw-elite-policy --no-viewer --fast --steps 320 --save elite-policy-smoke.rrd
python -m rerun rrd verify elite-qt-smoke.rrd elite-policy-smoke.rrd
```

## 显示性能与控制边界

Qt/control 线程只读取原生状态并发布不可变快照。显示端最多 30 Hz，保留一个
正在处理的帧和一个可覆盖的最新帧；慢 Viewer 不会让旧画面在 FIFO 中排队。
碰撞开关也交给显示线程。窗口显示合并帧数、显示处理耗时和错误；关闭会限时
等待，超时或显示错误会报告。Python 无法强杀卡死的显示回调。
Elite 流的 Rerun 出批间隔设为 33 ms，避免 SDK 默认 200 ms 缓冲造成额外拖动延迟。

Elite 显示使用父子相对变换，固定连接与声明工具变换只记录一次，未变化的姿态
和坐标轴不重复发送。碰撞包络按同一刚性父坐标和颜色合批：原有 351 个包络的
全部 8148 个三角面仍在，合为 24 次网格绘制，没有删减物理包络或简化碰撞模型。
开关沿用同一个视图 ID，尽量保留用户相机。

这些改动只作用于显示与接口。`EliteStationAdapter`、原生步长、原有
`EliteSandboxRuntime.poll` 调度、每步碰撞检查、阈值与控制限制保持原样。
显示帧可合并，录制是显示采样而不是每个控制周期的审计；原生运动不会因显示
采样而跳步。策略示例的等待只控制应用循环的墙钟节奏。

策略完整参考、提交回执、原生 planned 值和实际反馈保持分离。示例中的 planned
和 feedback 都明确标为仿真；没有实际发送回调时 sent 为 unavailable。
提交成功不代表已发送或到达。策略选择本身不执行控制，只有显式执行选中步骤
才通过与 Qt 相同的公共输入 API 提交。

## 开发与依赖

Python 3.12+；核心只依赖 NumPy。Qt、Rerun、网格加载与 Elite 均为可选功能，
导入核心不会导入这些运行时。推荐独立安装本包，并在应用侧使用薄适配器；
不要求 vendor 子模块，也不反转核心与机器人 SDK 的依赖。

当前测试固定使用 `rerun-sdk==0.38.1`、Trimesh 5.1.1、PyCollada 0.9.3，
请使用匹配的 Viewer。全新项目环境可按以下顺序安装：

```sh
python -m venv .venv
.venv/bin/python -m pip install evidence/wheels/libpyelite-0.1.0-cp312-cp312-linux_x86_64.whl
.venv/bin/python -m pip install -e '.[elite,rerun,qt,test]'
```

保留的当前生产源码 wheel 的 SHA-256 为
`dea977961be77d128cf47f8b077ccb6cd1cec88bada0e00a69f55eb8275b34d6`，
源码与构建依据见 `evidence/phase1-implementation.json`。同为 0.1.0 的旧 wheel
可能含旧规划器；此 wheel 适用于 CPython 3.12/Linux x86-64，其他平台需匹配的
SDK 安装。本次未重新编译原生库，也未修改原始 SDK 工作区。

`--station diagonal-07|diagonal-08`、`--recipe path.yaml` 与原有 recipe TCP 配置
继续可用；站点与 recipe 不匹配时会拒绝。`--collision-profile auto|off|path.json`
保留 SDK 语义，`off` 是用户显式选择，不是显示优化的一部分。
`rmw-elite-qt --ui generic` 仍提供通用元数据输入面板。

验证命令与本次性能证据：

```sh
python -m pytest -q
python -m pip check
```

`evidence/elite-performance/implementation.json` 保存真实 Qt 信号、原生轨迹对照、
慢发送与碰撞开关前后指标、持续 Viewer 运行日志、RRD 和截图。
既有 Phase 1–4 证据和 0.4.0 分发 wheel 均保留。本次是本地 0.4.1 显示修复，
没有重新进行全量分发验收，也没有硬件验证或硬实时保证。

## 公共接口与其他参考示例

`RobotDescription` 声明 URDF、命名关节、包资源根、基座与工具；`RobotState`
提供完整命名关节与可选 TCP/目标。关节使用弧度或米，距离使用米；刚性 4×4
变换为作用于列向量的 `parent_from_child`，TCP 目标与观察使用世界坐标。

`RobotSession.activate(source)` 取得唯一输入权；Qt 与 `PolicyInput` 都通过同一
`InputSession.submit` 提交。显式 `replace=True` 才转交输入权，关闭输入不取消
机器人已接受的目标。SDK 的连接、启停、模式、线程调度、IK 与安全均由调用方
拥有；渲染与回放不获得控制对象。详见 [适配器说明](docs/adapters.md) 与
[策略数据语义](docs/policy.md)。应用可配置自己的 `rerun.RecordingStream`，
交给公开 `RerunRenderer`；流、Viewer 与渲染线程生命周期仍由应用负责。

已有便携示例 `rmw-demo`、`rmw-callback-demo` 和外部 URDF/Flexiv 薄适配器保持
可用。wheel 内的七轴原始几何 fixture 为 CC0 示例，不是校准机器人。
真实 Rizon4 仍需显式 `--urdf` 与配套资源，资产来自 Flexiv 官方固定提交
`8b8452105c5faf088f456db18b0e2581b16fc630`（Apache-2.0），未复制进 wheel。
Flexiv 注入适配器只映射调用方已配置对象，不启停控制；源时间戳与实际发送证据
缺失时保持 unavailable，主机轮询时间不冒充源采样时间。
`rmw-demo-qt` 仍只打开输入面板，`--save` 录制后关闭面板，再用
`rmw-policy-replay` 回放；该通用示例不自动连接实时 Viewer。
