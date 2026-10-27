# Robot Motion Workbench

Elite CS66 双臂的离线交互与策略参考展示。输入使用采集数据动作参考或手写目标，运动由已有
`libpyelite.sim.StationSimulation`、原生规划器与碰撞检查执行；不运行 VLA 推理，
也不连接或控制真机。

## 启动环境

在本项目目录使用项目自己的环境，每个新终端先激活一次：

```sh
source .venv/bin/activate
```

当前环境已包含原生 Elite wheel、Qt 和固定版本的 Rerun。不要混用另一个 Conda
环境中的入口。更新本项目源码后的本地入口只需 `python -m pip install --no-deps -e .`。
全新环境的安装说明见下方“开发与依赖”。

## 快速开始

```sh
rmw-elite-qt --collision-profile auto
rmw-elite-policy
```

第一个打开原有的关节与 XYZ/RPY 输入面板和独立 Rerun Viewer。拖动提交的是目标，
机器人按原生速度、加速度与碰撞约束运动；不是瞬移。碰撞包络开关只改变显示，
不会关闭碰撞检查。目标不可达或安全检查锁存时，按面板说明显式重置。
Joint 页左右臂的 J6 下方、IK 页各自的 yaw 下方均有夹爪开度滑条，两页目标数值同步。
0% 为闭合、100% 为张开；旁边单独显示仿真实际开度。
操作夹爪滑条时，Rerun 回到当前仿真时间并继续实时播放，避免停留在暂停回看的旧画面。
夹爪与机械臂共用输入所有权，切换到 policy 后旧 Qt 输入失效。

第二个默认打开 Viewer，按原始时间戳同步播放采集 episode 0 的双臂 TCP 动作参考，
共 4451 帧、30 Hz、约 148.37 秒，播放一次后结束，同时播放左右夹爪开合动作。
蓝色是当前 chunk 从选中点起尚未播放的 TCP 参考路径；橙点是当前选中的目标 TCP，
不表示已到达。绿色是仿真 TCP 历史轨迹，绿色端点是最新反馈位置，机器人模型显示
仿真实际状态；它们在**同一个 3D 视图**内。没有单独的轨迹视窗。
TCP 点取配置工具坐标系在世界坐标系中的 XYZ，不是某一根手指的尖端；开合不改变 TCP 定义。
目标点不显示 `left selected` / `right selected` 文字，避免遮挡轨迹。
`--telemetry` 可增加文本状态栏，`--duration 12` 可只播放前 12 秒。
`--arm both` 是默认值；每帧先提交左右臂 TCP 和夹爪目标，再统一推进一次原生站点仿真，
沿用双臂碰撞协调。两臂分别显示参考路径、选中点和反馈轨迹，共用一个机器人模型与 3D 视图。
`--arm left` / `--arm right` 可切换为单臂播放，此时另一臂保持启动姿态。
数据没有关节角，`--kind joint` 继续使用手写关节参考；`--source synthetic`
可显式使用原来的约 12 秒 TCP 示例。
Viewer 侧栏和时间栏使用 Rerun 默认显示，不自动收起。

采集任务为“打开蛋盒，将鸡蛋逐个移到蛋托”。样例来自
`/home/olorin/datasets/remote/data/chunk-000/file-000.parquet`，使用 `action`
列中的世界坐标 XYZ（米）和 6D 姿态（旋转矩阵的前两列），不做缩放、平移或循环。
原始动作、帧号、时间戳和来源 SHA-256 保存于包内的
`assets/elite_episode_000.npz`（约 174 KiB），运行不依赖挂载路径或视频。
数据集的目标作为 `reference`；机器人姿态和绿色轨迹仍为原生仿真的 `simulated`
反馈，不用采集的 `observation.state` 冒充。`left_gripper.pos` / `right_gripper.pos` 的
0..1 开度按站点配置映射到 URDF 手指关节，按 URDF 速度限制进行离线运动模拟；
mimic 手指随动，碰撞检查与显示使用相同手指位置。夹爪不是电机动力学或抓取接触仿真。
本 episode 左夹爪范围 0..100%，右夹爪约 81.6..100%，右侧动作幅度较小。
是否能跟踪目标仍取决于当前站点、工具与碰撞约束；
原始数据的参考路径与仿真实际路径可能不同。

如需换另一条采集 episode，可导出后指定文件（读取 Parquet 需要 Rerun 环境中的 PyArrow）：

```sh
python scripts/import_elite_episode.py /home/olorin/datasets/remote \
  --episode 1 --output /tmp/elite-episode-1.npz
rmw-elite-policy --trajectory /tmp/elite-episode-1.npz
```

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

本仓库没有 `third_party` 或 Git 子模块。Elite 通过已安装的 `libpyelite` Python
包加载：Qt 复用它的输入窗口，Qt 与 policy 的目标都经 `EliteStationAdapter`
交给 `StationSimulation` 及其原生规划器、IK 和碰撞检查；Rerun 从该包的
`assets` 读取 URDF 与网格。本项目只负责输入适配、状态传递与显示。
`rmw-elite-policy` 默认使用采集数据中的动作作为策略参考，尚未接入实际 VLA 推理。

在项目环境中可检查 SDK 和原生库的实际位置：

```sh
python -c 'import libpyelite; print(libpyelite.__file__); print(libpyelite._native.__file__)'
```

当前测试固定使用 `rerun-sdk==0.38.1`、Trimesh 5.1.1、PyCollada 0.9.3，
请使用匹配的 Viewer。全新项目环境可按以下顺序安装：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install vendor/wheels/libpyelite-0.1.0-cp312-cp312-linux_x86_64.whl
.venv/bin/python -m pip install -e '.[elite,rerun,qt,test]'
```

保留的当前生产源码 wheel 的 SHA-256 为
`dea977961be77d128cf47f8b077ccb6cd1cec88bada0e00a69f55eb8275b34d6`，
源码提交、构建方式与校验信息见 [Elite SDK 来源](vendor/libpyelite-provenance.json)。
同为 0.1.0 的旧 wheel 可能含旧规划器；此 wheel 适用于 CPython 3.12/Linux x86-64，其他平台需匹配的
SDK 安装。`vendor/wheels` 只保存本机安装包，wheel 不纳入 Git；新克隆需另行取得
匹配的 wheel 或从记录的 SDK 源码构建，不能仅凭版本号安装任意 0.1.0 包。

`--station diagonal-07|diagonal-08`、`--recipe path.yaml` 与原有 recipe TCP 配置
继续可用；站点与 recipe 不匹配时会拒绝。`--collision-profile auto|off|path.json`
保留 SDK 语义，`off` 是用户显式选择，不是显示优化的一部分。
`rmw-elite-qt --ui generic` 仍提供通用元数据输入面板。

验证命令：

```sh
python -m pytest -q
python -m pip check
```

`evidence/` 是历史验证产物（重复虚拟环境、日志、截图与录制），不参与运行，
其中需要保留的 Elite 安装 wheel 已移到 `vendor/wheels/`；原 `evidence/` 目录现已清理。
`build/`、`.pytest_cache/`、`__pycache__/` 和示例生成的 `.rrd`/结果 JSON
也可清理。`.venv/` 包含实际运行依赖，应保留；它与生成文件都不纳入 Git。
保留 `tests/` 和 `scripts/` 用于验证控制、适配与显示行为。离线测试不代表硬件
验证或硬实时保证。

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
