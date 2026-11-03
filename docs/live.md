# 真实 Policy、图像与机器人状态显示

在线显示只消费宿主提供的数据。机器人控制、SDK 连接、相机采集和策略推理由宿主管理。
原来的 Qt 与离线 Policy 工具保留。

## 独立显示进程

`xense_motion_viewer.live_process.LiveDisplayProcess` 使用 `spawn` 启动子进程。
Rerun 导入、RecordingStream、URDF 渲染、图像缩放和 JPEG 编码、日志与录制都在子进程中运行。
宿主继续独立采样 SDK 状态，通过 `publish()` 提交不可变 `LiveFrame`；图像不改变这条状态来源。

`publish_images()` 只限频并保存最新图像引用。后台搬运线程完成序列化和复制，再写入固定容量的跨进程通道。
每路图像保留最新待处理样本，运动数据保持有界 FIFO 和批处理，任务使用完整状态快照及最多 256 条事件。
通道锁忙或运动队列满时丢弃显示数据，不等待显示消费。`stats` 提供错误、丢弃和进程存活状态。
独立进程仍共享 CPU 和内存带宽，不能视为零性能开销。

发布的图像必须是 HWC、uint8、RGB。调用者交付后不得原地修改或复用数组存储。
当前 Elite 腕部与 RECTIFY 驱动每次发布新的处理后 RGB 数组，符合这个约定；其他设备接入时必须重新确认。
显示只处理副本，不会在原图上画标记或缩放。

图像默认 10 Hz、最长边 640 像素、JPEG 质量 85。默认每路传输槽上限 8 MiB，超限丢弃并计入错误。
只有 `ImageStream` 中声明的图像才会显示。没有 Viewer 且没有录制输出的配置会被拒绝，避免写入无限缓冲。
输出每秒执行有时限的 flush；输出阻塞超过限时会结束显示进程，宿主继续运行。此时录制也随显示结束，
已写文件不再增加数据。重新开启显示需要新会话。

关闭会先限时排空搬运和显示通道，再退出子进程；超时可终止显示进程并返回 `False`。
强制退出可能留下未完成的录制文件，必须检查返回值。显示关闭不关闭宿主的 SDK 或相机。

## 数据和时间

现有运动输入保持如下语义：

- `FutureTrajectory(step_ids, poses)` 是剩余 TCP 参考，世界坐标位姿为 `[N,4,4]`。动作消费移除前缀，替换或混合更新有效参考。
- `LiveFrame(time, futures, state, sample_id, reset, status)` 是不可变运动快照。缺少实测状态时传 `None`，重复 `sample_id` 不重复累积历史。
- 蓝色轨迹表示未来参考；绿色路径和模型来自 SDK 实测 TCP 与关节角。提交动作不代表已到达。
- `generation` 随 episode 递增。运动、图像和任务都携带它，迟到的旧轮数据被丢弃；不依赖一个可能丢失的 reset 消息清除旧画面。

`clock_origin` 是宿主唯一的 `time.monotonic()` 起点。所有图像、状态和任务时间都减去这个起点，记录在 `host_elapsed`。
图像时间是原观测的采样时刻，不是硬件曝光时间。观测流超过 `stale_after` 未更新时清空图像并标记过期；
驱动不断返回旧缓存时无法据此判断单个传感器是否仍产生新帧。

机器人位姿使用米、右手坐标系和 `world_from_tcp`，关节角使用弧度。宿主负责对齐 URDF、世界坐标和工具变换。

```python
import time
from xense_motion_viewer.live import LiveFrame
from xense_motion_viewer.live_process import (
    ImageStream, LiveDisplayProcess, ProcessDisplayConfig, TaskSnapshot,
)

# 包含 spawn 的应用入口需要 if __name__ == '__main__' 保护。
origin = time.monotonic()
display = LiveDisplayProcess(
    description,
    config=ProcessDisplayConfig(spawn_viewer=False, save_path='recordings/live.rrd'),
    streams=(ImageStream('wrist', '腕部 RGB'),),
    clock_origin=origin,
)
try:
    at = time.monotonic() - origin
    display.publish(LiveFrame(at, state=measured_state, sample_id=sdk_sample_id), generation=1)
    display.publish_images(at, {'wrist': stable_rgb_array}, generation=1)
    display.update_task(TaskSnapshot(at, generation=1, episode=1,
                                    instruction=verified_instruction, source=instruction_source))
finally:
    if not display.close():
        raise RuntimeError('Display did not close gracefully')
```

原 `LiveMotionDisplay` 是进程内消费器，继续供已有本地使用方与录制测试使用；OpenPI 在线入口统一使用 `LiveDisplayProcess`。

## 任务区域

Rerun 0.26.2 默认字体不包含中文，面板标签、占位及运行事件采用英文。真实策略指令原样记录，不翻译。

Current Task 显示宿主提供的真实指令及来源、运行状态和 episode。
没有可信指令时显示 `Task instruction unavailable`；Sub task 和 Current action 固定显示 `Not connected`。
Task Timeline 是运行事件列表，不推测语义阶段、成功率或任务完成百分比。

宿主在状态变化时提交 `TaskSnapshot`，携带递增 ID 的 `TaskEvent` 历史。重复快照不会重复写入事件。
宿主仅保留最近 256 条；显示消费者检测到实际事件序号跳跃时标记缺口。录制是用于显示的采样记录，不是逐控制周期审计或完整图像数据集。

## xense-openpi 接入

`VisualizationSession` 保留独立 SDK 状态采样线程，Elite 适配器仍直接读取 RTSI 与夹爪缓存，不调用 `get_observation()`。
图像从原 `raw_obs` 在模型筛选之前取出，主路径只发布引用，不再次读取相机。
`enable_tactile` 继续决定设备采集；显示开关不改设备开关、`robot.cameras` 或连接判定。
关闭触觉时其面板显示未启用或尚无观测。模型输入仍是原 state、head 与双腕。

两种 runtime 的已有 Subscriber 接口提供 episode 和运行事件。环境包装器提供 reset、观测异常和动作提交异常，
并保留原错误传播。最终 reset 不会被计为新 episode；运行结束不代表任务成功。Dry run 会标明未发送机器人动作。

当前 Elite 请求不提供 prompt，因此读取服务端握手中带 `request_has_no_prompt` 条件的默认指令元数据。
该值来自实际输入变换中最先有效的 `InjectDefaultPrompt`；不会修改模型输入或注入优先级。
如果其他接入方增加请求级 prompt 或改写 prompt 的自定义变换，必须另外提供对应请求的真实指令。

默认关闭显示时，不创建显示进程、搬运线程或 SDK 显示采样，不加载 Rerun。
SDK 读取超过关闭等待时限会报告异常并阻止重叠启动第二个采样线程。
动作与状态的既有只读订阅接口保留；RTC 和 PacedBroker 的参考语义不变。

Rerun 继续固定 0.26.2。开发联调应将运行环境的 editable XMV 指向本次新分支；发布时需同步 OpenPI 的子模块固定提交。
