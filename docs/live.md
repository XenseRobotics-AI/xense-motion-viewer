# 真实 Policy 与机器人状态显示

XMV 的在线入口只消费数据，不需要控制会话，也不会调用 IK、推进仿真或发送机器人命令。
原来的 `xmv-elite-policy` 和 Qt 离线工具保留。

## 数据入口

`xense_motion_viewer.live` 提供：

- `FutureTrajectory(step_ids, poses)`：某个 TCP 的剩余参考，位姿为世界坐标 `[N,4,4]`。
  ID 在当前 episode 内唯一；动作消费移除前缀，替换/混合则更新有效参考。
- `LiveFrame(time, futures, state, sample_id, reset, status)`：一个不可变显示快照。
  `time` 是宿主单调经过时间，`state` 是 SDK 实测 `RobotState`；缺失状态传 `None`。
  同一个 `sample_id` 不重复累积历史。`reset=True` 清空上一 episode 的参考和历史。
- `LiveMotionDisplay(recording, description)`：同进程、有界异步渲染，渲染线程只接收不可变数据。

蓝色路径与点表示未来 TCP 参考。绿色路径与点来自实测 TCP；机器人各连杆由实测关节角和
URDF 正运动学更新。未来参考从不用于生成实测历史。单个点的未来参考也可以显示。
历史使用有界分段存储；数据缺失时保留旧线段并隐藏当前点，恢复后另起一段。

位姿使用米、右手坐标系和 `world_from_tcp` 方向；关节角使用弧度。宿主负责对齐 URDF、
policy 世界坐标、安装变换与工具变换。`poses_from_xyz_rotation6d()` 将 `[N,9]` 的 XYZ 和
旋转矩阵前两列转换为 `[N,4,4]`，进行正交化并拒绝退化姿态。

## Rerun 版本与显示线程

当前分支固定 Rerun 0.26.2，适配 xense-openpi 的 lerobot 版本约束。显示与推理共用 Python
环境，状态采样和渲染各自运行在独立线程中；渲染线程只接收状态和参考的副本。
0.26 使用实体树解析坐标关系：网格放在对应连杆实体下面，连杆由世界位姿更新；碰撞包络与
TCP 标记也使用世界位姿。新版命名坐标系 API 不作为 0.26 正确显示的前提。
0.26 的相机取景和时间跟随由 Viewer 控制；新版 SDK 的相机初始位姿与播放状态 RPC 不在
该版本的 Python API 中。打开/重连 Viewer 不会替换已有录制文件的输出。

```python
from xense_motion_viewer.live import LiveFrame, FutureTrajectory
from xense_motion_viewer.live import LiveMotionDisplay
import rerun as rr

recording = rr.RecordingStream("live")
recording.save("recordings/live.rrd")
display = LiveMotionDisplay(recording, description)
try:
    display.publish(LiveFrame(
        time=host_elapsed,
        futures={"left": FutureTrajectory(action_ids, world_from_tcp_poses)},
        state=measured_state,
        sample_id=sdk_sample_id,
    ))
finally:
    display.close()
```

`publish()` 不等待渲染；队列满会拒收并计数。应检查 `stats.errors`、
`dropped_full` 和关闭返回值。显示采样与录制不是逐控制周期审计。
关闭操作限时排空队列、退出显示线程并断开 Rerun。

## xense-openpi 侧的边界

公共 `ActionStream` 接收完整 chunk、调度后有效队列和提交进度，不包含 Elite 或 20 维假设。
`VisualizationSession` 管理这些订阅、独立 SDK 状态采样及 XMV 显示线程。
每种机械臂实现一个 `RobotVisualizationAdapter`：提供静态模型描述、动作到 TCP 位姿的解码、
以及 SDK 实测状态。Elite 的维度切片、RTSI 访问与夹爪标定均留在 Elite 私有适配中。

普通模式的未来参考只包含 `action_horizon` 范围内的动作。RTC 显示原队列完成实际延迟裁剪
和混合后的参考。PacedBroker 已缓存的动作仍属于待提交参考，直到 runtime 成功向环境提交。
提交进度只缩短参考，不能证明机械臂已经到达；已走路径始终来自实测 TCP。

默认关闭可视化。关闭时不创建订阅、SDK 采样线程或显示线程，不加载 Rerun；推理与控制继续
使用现有 broker 和 SDK 连接。启用后的 `set_enabled(False)` 关闭订阅和显示资源；
SDK 读取若超过关闭等待时限仍阻塞，会报告异常，并阻止新采样线程与旧线程重叠。

xense-openpi 通过 Git submodule 固定本项目的提交。启用显示的 Elite 入口选择
`ObservedActionChunkBroker` / `ObservedRTCActionChunkBroker`，复用原有调度和队列计算；
原公共 broker、动作队列、PacedBroker 和 runtime 不需要为显示修改。
运行中关闭再开启时，显示会等待新计划，无法恢复已经留在 PacedBroker 中的全部旧缓存。
需要完整剩余参考时，应在 episode 开始前启用，并保持到该 episode 结束。
