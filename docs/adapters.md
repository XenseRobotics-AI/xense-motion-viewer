# Elite 输入、状态与运行时

`EliteStationAdapter` 将 `libpyelite.sim.StationSimulation` 接到工作台。
Qt 与 Policy 共用 `RobotSession` 和目标提交接口；原生 SDK 负责 IK、规划、
速度与加速度限制、碰撞检查及错误锁存。

## 模型与目标

`RobotDescription` 提供 SDK 自带的 URDF、命名关节、基座、工具和碰撞几何。
`RobotState` 提供不可变的完整关节状态及 TCP 位姿。位置使用米，角度使用弧度；
4×4 刚性变换采用作用于列向量的 `parent_from_child` 约定。

适配器提供四个关节组：`left`、`right`、`left_gripper`、`right_gripper`，
以及 `left`、`right` 两个 TCP 目标。机械臂命名关节映射到 SDK 的关节顺序；
夹爪开度映射到站点配置的独立手指关节，mimic 手指由 URDF 推导。

```python
from libpyelite.sim import StationSimulation
from robot_motion_workbench import RobotSession, TcpTarget
from robot_motion_workbench.adapters.elite import EliteStationAdapter

simulation = StationSimulation('diagonal-08', collision_profile='auto')
adapter = EliteStationAdapter(simulation)
session = RobotSession(adapter)
inputs = session.activate('application')

pose = session.snapshot().tcp_poses['left'].copy()
pose[0, 3] += 0.001
receipt = inputs.submit(TcpTarget('left', pose))
adapter.step()
state = session.snapshot()
inputs.close()
```

`JointTarget(group, positions)` 必须完整提供该组命名关节。
`TcpTarget(tcp, world_from_tcp)` 使用世界坐标刚性位姿。
会话在提交前检查名称、有限值和静态范围，不裁剪或补齐目标。
可达性和动态约束仍由原生 SDK 判断。

`SubmissionReceipt.status == 'submitted'` 仅表示适配器调用成功。
目标提交不会推进仿真，也不会把目标值写成反馈；反馈来自后续 `snapshot()`。

## 输入权与 Qt

同一会话只有一个输入来源持有输入权。`session.activate('policy', replace=True)`
显式接管后，旧 Qt 输入无法提交或重置机器人；关闭旧输入也不会释放新输入的所有权。
释放输入权不等于取消 SDK 已接受的目标。

`rmw-elite-qt` 复用 SDK 的 Joint / IK 面板，输入通过会话转发。
两页夹爪目标同步；观察刷新不会再次提交目标。夹爪输入还会让 Viewer 返回当前仿真时间。

SDK 调用在宿主线程执行。`EliteSandboxRuntime.poll()` 沿用原生沙盒的有限追赶调度，
逐步调用 `StationSimulation.step()`。碰撞检查失败时保留原生锁存行为，需显式重置。

## 显示与录制

`EliteRerunView` 使用独立显示线程，默认最多 30 Hz，可用 `--display-hz 60` 调整。
显示间隔从一帧开始处理时计算，处理耗时计入间隔；只保留正在处理和最新待显示状态。
慢渲染不会积压旧画面或阻塞目标提交。静态模型、父子变换与碰撞网格按需记录，
碰撞包络显示开关不改变原生碰撞检查。

应用同时持有实时 Rerun 服务和可选文件 sink。关闭 Viewer 不影响仿真或录制，
重连补发场景快照；已连接时再次打开不会重置相机。显示错误和关闭超时通过统计及退出状态报告。
录制包含显示采样，不保证保存每个原生控制周期。
