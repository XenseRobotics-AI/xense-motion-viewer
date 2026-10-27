# Robot Motion Workbench

Elite CS66 双臂离线运动工作台，使用原生 `libpyelite` 仿真、Qt 输入面板和 Rerun 可视化。

当前维护两条使用路径：

- **Elite + Qt**：交互设置双臂关节、TCP 位姿与夹爪开度，查看仿真运动和碰撞包络。
- **Elite + Policy**：播放采集动作或手写参考，在同一 3D 视图中对比目标与仿真反馈。

支持 `diagonal-07`、`diagonal-08` 两种站点，以及 RRD 录制、回放和 Viewer 重连。运动规划、IK 与碰撞检查由 SDK 负责；当前不连接真机，不运行 VLA 推理。

## 安装

需要 Python 3.12+。当前验证环境为 **CPython 3.12 / Linux x86-64**，使用与 [SDK 来源记录](vendor/libpyelite-provenance.json) 匹配的 `libpyelite` wheel。

wheel 不随 Git 仓库分发，需单独取得或按来源记录构建；同为 `0.1.0` 的其他构建不保证兼容。在仓库根目录执行：

```sh
python3.12 -m venv .venv
source .venv/bin/activate

# 替换为已取得的匹配 wheel 路径
python -m pip install /path/to/libpyelite-0.1.0-cp312-cp312-linux_x86_64.whl
python -m pip install -e '.[elite,rerun,qt]'
```

Rerun SDK 固定为 **0.38.1**，请使用同环境附带的 Viewer。已有环境只需激活 `.venv`；更新本项目源码后的入口可用 `python -m pip install --no-deps -e .` 刷新。

## 快速开始

```sh
# Qt 交互面板与实时 Rerun Viewer
rmw-elite-qt --collision-profile auto

# 双臂策略参考与仿真反馈
rmw-elite-policy
```

Qt 提交运动目标，由原生仿真按速度、加速度与碰撞约束推进。Joint / IK 页的夹爪输入同步，0% 为闭合、100% 为张开，实际开度单独显示。碰撞包络开关只改变显示；`--collision-profile off` 才会关闭碰撞检查。

Policy 默认播放包内 episode 0：4451 帧、30 Hz、约 148 秒，左右臂 TCP 与夹爪按原始时间戳同步播放一次，无需数据挂载或视频。蓝线是待提交参考，已提交部分隐藏，橙点是选中目标，绿线与机器人模型是仿真反馈。夹爪模拟开合运动，不模拟物体抓取接触。

| 常用参数 | 用途 |
| --- | --- |
| `--station diagonal-07` / `--station diagonal-08` | 选择站点；未提供 recipe 时默认 `diagonal-08` |
| `--recipe path.yaml` | 使用 SDK 的站点与工具配置，须与显式指定的站点一致 |
| `--display-hz 30` / `--display-hz 60` | Qt / Policy 目标显示频率，默认 30 Hz；不改变动作输入频率或原生仿真步长 |
| `--arm left` / `--arm right` | Policy 只播放单臂；默认 `both` |
| `--duration 12` | Policy 只播放前 12 秒 |
| `--source synthetic` / `--kind joint` | Policy 使用手写 TCP / 关节参考；采集数据没有关节角 |
| `--trajectory path.npz` | Policy 使用另一条已导出的采集轨迹 |
| `--telemetry` | Policy 增加文本状态栏 |

更多选项见各命令的 `--help`。

## 录制与回放

```sh
rmw-elite-qt --collision-profile auto --save recordings/elite-qt.rrd
rmw-elite-policy --save recordings/elite-policy.rrd

# 结束对应应用、完成写入后回放
rmw-policy-replay recordings/elite-policy.rrd
```

关闭 Viewer 不会停止仿真或录制；Qt 的“打开 Rerun”按钮可重连。Qt / Policy 默认 Viewer 端口为 9876 / 9877，可用 `--port` 修改。`--no-viewer` 禁用实时 Viewer，仍可录制。

无需界面的有限步数验证：

```sh
rmw-elite-policy --no-viewer --fast --steps 320 --save recordings/smoke.rrd
python -m rerun rrd verify recordings/smoke.rrd
```

`rmw-elite-record` 可单独录制原生仿真状态。录制采用显示采样，可能合并或丢弃显示帧，不是逐控制周期的完整记录；提交成功也不代表已发送或已到达。

## 开发与文档

安装上述运行依赖后执行：

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m pip check
```

测试覆盖公共输入与状态、原生规划和碰撞、夹爪、双臂轨迹、RRD 内容、Qt 交互和显示线程。测试中的小型 URDF 仅用于验证变换与网格计算，不提供独立演示入口。

- [Elite 输入、状态与运行时](docs/adapters.md)
- [Policy 数据、轨迹导入与回放语义](docs/policy.md)
- [Elite SDK 来源与构建信息](vendor/libpyelite-provenance.json)
