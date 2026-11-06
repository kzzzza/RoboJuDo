# G1 工作站直连部署

工作站运行摄像头、YOLO/HMR2/TCAM、TCN（或 Original/MLP）、UFO Actor 和控制状态机。工作站通过官方 DDS 订阅 G1 反馈、发送关节目标，机器人端无需安装自定义模型或进程。

```text
工作站视觉进程（GPU） ── 本机 Unix socket：z[256] ── 工作站控制进程（Actor CPU，50Hz）
                                                        │ rt/lowcmd
                                                        ▼
                                                       G1
                                                        │ rt/lowstate
                                                        └── 关节、IMU、遥控器反馈
```

配置为 `g1_ufo_video_real_workstation`，使用独立的 `UnitreeG1DdsEnv`。原有 `g1_ufo_video_real_stream` 仍是 Jetson/SSH 方案，两者可以并存。

## 1. 环境与模型

在仓库根目录操作，要求 Linux、Python ≥3.11。使用现有 `.venv` 控制环境和 `.venv-video` 视觉环境；依赖见项目 `requirements.txt` 和 [视频开发说明](ufo_video.md)。控制端使用 CPU ONNX Runtime；视觉端默认 `cuda:0`。

安装官方 SDK 的本地依赖：

```bash
bash scripts/setup_unitree_workstation.sh .venv/bin/python
```

脚本需要 Git、CMake、C/C++ 编译器及网络访问。SDK 固定到 `814556d15970dd2ecf1c9984e845ca02ab07e206`，CycloneDDS 固定到 0.10.2。源码、原生库和 pip 缓存位于 `.local/`，Python 包安装在指定的项目虚拟环境，不使用 sudo，也不修改基础 Conda 环境。不要复制其他机器或其他目录的虚拟环境。

模型沿用当前本地模型包：

- `.local/rgb_bfm/original_pipeline/`：共享视觉前端及 Original 所需模型。
- `.local/rgb_bfm/tcn_pipeline/` 或 `mlp_pipeline/`：相应蒸馏头。
- `assets/models/g1/ufo_m0/FBcprAuxModel.onnx`、元数据及哈希清单：控制端 Actor。

本配置不需要 LAFAN 动作或控制端 backward encoder。模型、缓存和运行输出不提交到 Git，也不写入 `my_data`、`my_models` 或 `my_modle`。

## 2. 有线与无线网络

`--net-if` 必须是**工作站上连接机器人网络的网卡名**，例如有线的 `enp3s0` 或无线的 `wlp2s0`；不要照抄 Jetson 的 `eth0`。通过 `ip -br address` 查看。

有线：连接 G1 交换机，按照官方网络配置给工作站网卡设置不冲突的 `192.168.123.X/24` 地址，常用 `192.168.123.99`。官方运控计算机通常为 `192.168.123.161`。

无线：工作站必须能通过所选 Wi-Fi 网卡与 G1 的 DDS 服务直接双向通信。相同 SSID 或 ping 成功不能证明 DDS 已通；AP 客户端隔离、组播过滤、路由和防火墙均可能影响发现。使用下面的只读预检确认。失败时程序报错，不切换网卡、不改系统路由、不自动启用 SSH 或机器人端中转。

两种连接使用同一 DDS 数据格式，默认 domain 0。不得同时启动另一套向同一 G1 发布底层指令的控制程序。

## 3. 预检

仅检查模型合同和哈希，不连接 G1：

```bash
.venv/bin/python scripts/check_ufo_m0_real.py \
  --config g1_ufo_video_real_workstation --latent-variant tcn --offline
```

检查 SDK 和工作站网卡，但不连接 DDS：

```bash
.venv/bin/python scripts/check_ufo_m0_real.py \
  --config g1_ufo_video_real_workstation --latent-variant tcn --net-if enp3s0
```

接收真实 G1 状态，确认递增 tick、CRC、29 关节、IMU 和遥控器数据格式：

```bash
.venv/bin/python scripts/check_ufo_m0_real.py \
  --config g1_ufo_video_real_workstation --latent-variant tcn \
  --net-if enp3s0 --read-only
```

只读检查不会创建 LowCmd 发布器，不释放内置运控，不发送动作，也不要求设置 `UFO_REAL_ROBOT_OK`。只读通信通过仍不等于底层动作和无线可靠性已经验收。

## 4. 启动与操作

此模型只用于 **29DOF、腰部解锁**的 G1。当前白名单中的 PR 反馈机型 ID 为 5（`g1_29dof_rev_1_0`）或 2（旧版 `g1_29dof`）；23DOF、锁腰、AB 模式、未知机型或电机故障会被拒绝。官方还列有 11/13/15/18 等新版硬件 ID，本实现先拒绝，需单独核对现有策略对应的机械模型与电机参数后再扩展。机型字段来自反馈，不能靠修改为某个数值绕过硬件不匹配。机型表见[官方 G1 模型说明](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md)。

保持 G1 吊装支撑，确认机器人遥控器可用，再运行：

```bash
UFO_REAL_ROBOT_OK=1 .venv/bin/python scripts/run_ufo_workstation_real.py \
  --net-if enp3s0 --variant tcn --camera-backend opencv --camera 0 \
  --show-detection-box
```

无线时将 `enp3s0` 替换为实际无线网卡名；Azure Kinect 改为 `--camera-backend k4a`，并提前安装 libk4a。无桌面时加 `--no-preview`，同时去掉 `--show-detection-box`。内置摄像头需要打开隐私开关、滑盖并保持全身入画。

启动器先校验所需视觉包和依赖、Actor 以及只读 DDS 反馈，然后启动控制与视觉进程。控制进程会通过官方 MotionSwitcher 释放并确认内置运控退出；失败就结束。**启动后仍然不发送关节目标，直到操作员按 A。**

| 操作 | 行为 |
| --- | --- |
| A | 从实测关节角经过 500 个控制周期（配置 50Hz，约 10 秒）渐进到默认站姿 |
| B | 已站稳且 z 有效时开始跟随；提前按 B 不会在未来自动启动，需要重新短按 |
| X / 人物丢失 | 用约 5 秒回站姿；重新检测到人后仍需再次短按 B |
| R2 | 优先停止策略，并在反馈新鲜时保持实测关节位置；松开后仍锁定，需松开 A 后重新短按 A、完成站姿初始化再按 B |
| 通信/反馈/计算异常 | 故障锁定；网络恢复不自动恢复策略，排查后重启进程 |
| RGB 窗口 q/Esc / 终端 Ctrl+C | 先让控制进程尝试新鲜反馈下的位置保持，再结束进程、释放资源 |

如需分别启动两个本机进程，可使用同一 socket 路径：

```bash
UFO_REAL_ROBOT_OK=1 .venv/bin/python scripts/run_pipeline.py \
  -c g1_ufo_video_real_workstation --net-if enp3s0 \
  --latent-variant tcn --latent-socket /tmp/ufo_workstation.sock \
  --metrics-json .local/workstation_control.json

# 另一个本机终端
.venv-video/bin/python scripts/run_ufo_video_source.py \
  --camera 0 --camera-backend opencv --show-camera --variant tcn \
  --packages .local/rgb_bfm --device cuda:0 --socket /tmp/ufo_workstation.sock
```

## 5. 命令语义和故障行为

Actor 原始输出不是可直接发送的关节角。沿用现有处理：`clip(raw * 5, -5, 5)` → 逐关节 UFO 缩放 → 叠加默认角 → 按关节名映射 → 位置与变化率保护 → `LowCmd.motor_cmd[].q`。增益保持原模型配置，`dq/tau` 为零，未使用的槽位保持禁用。

驱动没有后台目标重发线程；每次显式控制 step 只发送一条命令。默认连接等待 5 秒；低层反馈年龄与当周期目标计算时长上限均为 100ms。CRC 错误、重复或倒退 tick 不刷新有效期；tick 的 32 位正常回绕允许。四元数转换为 RoboJuDo 所用 xyzw。

视觉潜变量继续使用 5 秒年龄限制，这是与低层反馈 100ms **不同**的检查。发生故障时不发送过期策略结果；如果仍有新鲜有效反馈，可尝试位置保持；反馈已过期则停止发布，不从旧反馈构造保持目标。完整断网时不能保证工作站的最后一条指令抵达机器人，机器人端最终行为取决于固件和电机超时配置，须另做现场验收。R2 是软件停止，不替代硬件急停。

## 6. 指标与验证

每次启动输出到 `.local/workstation_runs/run_*/`：

- `preflight.log`、`control.log`、`vision.log`：启动检查和故障原因。
- `control_metrics.json`：采集到控制发送前的延迟、控制计算时长、反馈年龄、发送计数、停止原因及故障状态。
- `vision_metrics.json`：取帧/检测/时序模型耗时、有效 z 更新率、人物丢失重置等。
- `summary.json`：进程状态、变体与网卡。

控制计算时间在主循环中测量；完整周期与反馈年龄另存于 `dds.samples`。延迟在工作站测量，不代表机器人实际执行或物理响应时间。时序样本使用有界缓存，计数保留累计值。视觉更新频率、逻辑模型帧数与 50Hz 控制配置不可混为一谈。

运行本机验证（需要 SDK，套接字必须在允许本机网络的环境中执行）：

```bash
PYTHONPATH="$PWD:$PWD/tests" .venv/bin/python -m pytest -q \
  tests/test_g1_workstation_dds.py tests/test_workstation_cli.py \
  tests/test_g1_workstation_loopback.py tests/test_ufo_video.py \
  tests/test_ufo_video_real.py tests/test_ufo_streaming.py \
  tests/test_ufo_m0_real.py tests/test_ufo_video_source.py
```

回环测试强制 domain 121、`lo`、关闭组播，仅发现 `127.0.0.1` 上的两个测试进程，不释放真实运控。它使用真实 HG 消息和 CRC、真实 Actor、原状态机及本机 z socket，验证 1000 个健康周期、A/B/R2、人物丢失、断流和恢复锁定；模拟端只模拟通信与关节反馈，不模拟动力学。摄像头到 MuJoCo 是单独的回归测试。以上均不构成真机动作、无线稳定性或断链行为认证。

### 本机验证记录（2026-11-06）

- 上述回归集：52 通过、1 跳过；跳过项依赖本机缺少的 LAFAN 动作文件。控制环境 `pip check` 通过。
- 单独保存的隔离 DDS 运行：1000 个健康跟随周期耗时 20.074 秒，计算时间 P95 为 5.60ms；模拟端累计收到 2260 条命令，CRC 与字段错误均为 0；断流后锁定、恢复不自动跟随。
- 本机摄像头 TCN → MuJoCo：2000 个循环、1409 次策略推理、179 个不同潜变量帧，未记录摔倒；采集到 PD 计算后延迟 P95 为 302.8ms。视觉有效发送约 6.32Hz，与 50Hz 控制循环不同。
- 视觉采用最新帧队列，该次运行丢弃了 158 个采集帧；有限步仿真退出、关闭 socket 后视觉进程结束前记录了 4 次发送失败。不得据此声称链路零丢帧或完成真实机器人延迟验收。
- 本机详细记录：`.local/setup/workstation-validation.json`、`workstation-tests.log`、`workstation-dds-loopback.json`；摄像头记录：`.local/camera_demo/run_20261106_181502_25745/`。
