# Policy 参考、轨迹与遥测

`rmw-elite-policy` 将采集动作或手写目标作为参考，驱动原生 Elite 双臂仿真。
它不运行模型推理；显示和回放都不取得机器人输入权。

## 参考与执行

`PolicyChunk` 保存完整参考序列、时间偏移、关节组、来源时间和过期时间，支持：

- `absolute_joint_positions`：命名关节的绝对位置。
- `absolute_tcp_poses`：世界坐标中的绝对 TCP 位姿。

`PolicyInput.accept()` 校验整个 chunk，`select()` 选择步骤；
只有 `execute_selected()` 通过已有 `InputSession.submit()` 提交目标。
时间偏移描述执行意图，实际调度由宿主循环负责。

替换或过期会清除当前选择与可视参考，不会取消 SDK 已接受的目标。
chunk ID 在同一 `PolicyInput` 生命周期内唯一，所有输入数组和事件均为不可变副本。
`StepRef` 标识 chunk 与步骤；重复执行同一步时，具体提交尝试由独立的回执序号区分。

## 内置轨迹与导入

默认数据来自包内 `assets/elite_episode_000.npz`：episode 0，4451 帧，30 Hz，
约 148.37 秒。读取 `action` 列中的世界 XYZ、6D 姿态和左右夹爪开度。
XYZ 单位为米，6D 姿态表示旋转矩阵的前两列，加载时正交化；不缩放、平移或循环播放。
原始动作、时间戳、帧号和来源文件 SHA-256 保存在 NPZ 中，运行不依赖原始挂载或视频。

默认 `--arm both`：左右臂选择同一源数据帧，先提交两臂 TCP 与夹爪目标，
再统一推进一次原生站点仿真，保留双臂碰撞协调。
`--arm left` / `--arm right` 只播放单臂，另一臂保持启动姿态。

每个参考 chunk 最多 60 帧，选择依据原始时间戳。`--duration N` 限制播放时长；
`--source synthetic` 使用手写 TCP 参考；`--kind joint` 使用手写关节参考，
因为采集数据不包含关节角。

导入另一条 Elite LeRobot v3 episode：

```sh
python scripts/import_elite_episode.py /path/to/dataset \
  --episode 1 --output recordings/episode-1.npz
rmw-elite-policy --trajectory recordings/episode-1.npz
```

导入需要 PyArrow，随 `rerun` 可选依赖安装。输入数据集须为 `v3.0` 格式，
`robot_type` 为 `bi_elite_cs66_rt`，包含所选机械臂的 TCP 和夹爪动作列。

夹爪开度范围为 0..1（闭合到张开），经站点配置转换为 URDF 手指关节，
按关节速度限制运动，并由原生碰撞检查验证每段位移；不安全时保持当前位置。
这只模拟开合运动，不包含电机动力学或物体接触力。

## 显示与数据语义

一个共享 3D 视图显示机器人、两臂参考和反馈：

- 蓝线：当前 chunk 尚未提交的 TCP 参考；已提交的点，以及两端均已提交的线段隐藏，不再另画提交历史。
- 橙点：选中的目标 TCP。
- 绿线及端点：原生仿真的 TCP 历史和最新反馈。

TCP 是配置工具坐标系的世界位置，不是某根手指的尖端；夹爪开合不改变 TCP 定义。
`--telemetry` 增加文本状态栏。

每个 chunk 的坐标只发送一次，进度更新只修改已提交部分的显示状态和橙点位置。
绿色历史采用最多 32 个小段循环保存，每段最多 64 点，相邻段共享端点；只更新正在追加的段，
不重复发送整条历史。机器人将变化的姿态批量发送；每条变换保留 Rerun 所需的父子坐标系标识，
网格只在初始化时发送。

`--display-hz 30`（默认）或 `--display-hz 60` 设置状态采样与显示批次频率，按仿真时间调度。
在跨过采样时刻的第一个原生步读取当前状态，结束时补齐最终状态；不改变动作提交和仿真步长。
`--fast` 仍按仿真时间采样，实际显示频率受显示线程和 Viewer 处理速度影响。

prediction、selection、submitted、planned、sent、feedback 分别存储。
采集动作标为 `reference`，原生规划值和反馈标为 `simulated`；
采集的 `observation.state` 不会被当成当前仿真反馈。
无实际发送回调时，`sent` 为 `unavailable`，提交成功不代表已发送或已到达。

`SampleTime.received_at` 是宿主收到数据的时间；`source_time` 与 `source_clock`
仅在存在源采样时间时成对提供。Elite 离线循环使用仿真经过时间作为宿主时钟，
并保留数据集或 `elite_simulation` 源时钟。不同时间基准不会被隐式相减。
反馈的 `state_time` 是显示时钟，不自动等于源采样时间。

原始事件按时间行写入固定的 `events/log` 实体，`event_sequence` 保留消费顺序，
避免长回放为每条事件创建新实体。文本记录按 0.25 秒或累计 256 条事件批量写入，
保留各自的原始时间，关闭时补齐尾部；这段等待不影响机器人姿态更新。
发送回调位于 `streams/sent`，自动生成的无发送证据记录
位于 `bookkeeping/sent`。后者不会覆盖已有发送回调，也不会替新一次提交证明发送成功。
反馈缺失时保留已有几何及其原始来源；来源在模拟和实测间变化时清空历史连线。

## 队列与回放

有界 `VisualizationWorker` 只消费显示数据，不能访问适配器或输入权。
Elite 按显示周期将参考变化、提交记录和当前状态一起入队，容量仍按事件条数计算。
显示线程按批处理事件，完整保留收到的原始记录和历史路径点，每批只绘制最新机器人姿态与目标标记。
若处理落后，多批可合并绘制，避免补画过时姿态；队列满时整批拒收并计数，不阻塞原生仿真。
应检查 `display_frames`、丢弃及错误计数和 `display_closed`；消费事件数不等于实际绘图帧数，
显示录制也不作为完整控制审计。

关闭时先释放输入，再限时排空显示队列、刷新并断开 Rerun。
`rmw-policy-replay recording.rrd` 只启动 Viewer，`replay_events()` 只消费不可变事件，
两者都不会执行录制中的目标。
