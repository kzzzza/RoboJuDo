# 工作站 AMO／UFO 双策略控制

AMO 负责零速度主动平衡和摇杆行走，UFO 负责视频动作跟随。两套模型均在工作站 CPU 上运行：AMO 使用现有 TorchScript 主策略与 adapter，UFO 使用现有 ONNX Actor。视觉进程继续使用 GPU。控制进程只有一个环境、一个 DDS 发布器，50Hz 显式发送命令，没有后台旧目标重发线程。

## 预检与启动

使用现有 `.venv` 和 `.venv-video`。AMO 权重位于 `assets/models/g1/amo/`，`artifact_manifest.json` 固定三个原有文件的 SHA256。未修改训练模型、权重或原策略参数。

仅验证两套模型、关节合同和 CPU 推理，不连接机器人：

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/check_ufo_m0_real.py \
  --config g1_ufo_video_real_workstation_amo --latent-variant tcn --offline
```

实际接收 G1 状态的只读预检（不创建 LowCmd 发布器，不释放运控）：

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/check_ufo_m0_real.py \
  --config g1_ufo_video_real_workstation_amo --latent-variant tcn \
  --net-if enp8s0 --read-only
```

完成吊装和现场检查后，操作员可用以下命令启动真实机器人控制：

```bash
UFO_REAL_ROBOT_OK=1 .venv/bin/python scripts/run_ufo_workstation_real.py \
  --with-amo --net-if enp8s0 --variant tcn \
  --camera-backend opencv --camera 0 --show-detection-box
```

Azure Kinect 改为 `--camera-backend k4a`，运行库安装见 [Kinect 说明](azure_kinect_setup.md)。未加 `--with-amo` 时仍使用原来的 UFO-only 流程，包括该配置原有的姿态检查设置；本轮未运行上述真机启动命令。

## 操作

| 操作／事件 | 结果 |
|---|---|
| 启动 | 两模型加载、预热，但不发送关节指令 |
| A | 3 秒从实测关节角过渡到 AMO 默认姿态，完成后等待 Y |
| Y | 0.5 秒进入 AMO；摇杆回中持续 0.5 秒后允许行走 |
| AMO 中 B | 有有效 z 才接受；先 0.5 秒降到零速度，再 0.5 秒切入 UFO |
| UFO 中 Y／X | 0.5 秒回零速度 AMO；回中确认后允许摇杆行走 |
| 人物丢失、z 过期或结束 | 反馈正常且 AMO 有效时，0.5 秒回零速度 AMO；恢复视觉后需重新短按 B |
| R2 | 当周期停止两套策略，反馈新鲜时保持实测位置；不改为纯阻尼 |
| 正常 R2 后恢复 | 松开相关按键，再短按 A → Y → B，不能直接 B／Y 重新启动 |
| 反馈、姿态、模型或发送故障 | 锁定停止；不能通过按键清除，排查后重启 |

R2 优先于 Y/X，Y/X 优先于 B，B 优先于 A。除 R2 按下状态外，操作均使用上升沿，不排队；例如初始化期间提前按住 Y，A 完成后不会自动启用 AMO。A 在行走或跟随中不重新初始化。

**A 及等待 Y 阶段仍是静态 PD，需要外部支撑；只有 Y 交接完成后 AMO 才主动平衡。** R2 是软件位置保持，不保证停止后能站稳，也不替代硬件急停。新 AMO 配置启用姿态检查，不支持原 UFO-only 配置允许的躺姿控制。

摇杆死区为 0.1。AMO 进入、视觉回退时先锁零速度，不继承切换前的摇杆指令。AMO 的 Y 内部切换事件不会收到状态机 Y；命令映射沿用 `G1AmoPolicyCfg`。

## 关节、增益与故障语义

- AMO 观测包含 23 关节，网络输出 15 个腿／腰关节。8 个手臂关节使用 AMO 默认角及其配套增益，6 个手腕使用现有 G1 默认角／增益；按名称补齐为 29 关节，不按数组尾部猜测。
- UFO 使用自己的缩放、默认角及配套增益。切换从最后实际发送目标及增益出发，用 0.5 秒平滑曲线过渡到目的策略实时目标；最终目标仍经过关节位置及变化率保护。
- 两策略的历史独立，切入时复位。AMO 完整清理历史、前动作、指令、步态，并用当前反馈初始化，不沿用原 `reset()` 只清计步器的行为。
- 保留 5 秒视觉年龄限制、100ms 低层反馈／当周期目标计算时限、CRC 与机型检查。视觉失效回 AMO 不清除硬件故障，也不自动恢复 UFO。
- 新模式下视觉进程异常退出，启动器保留控制进程，控制器在视觉数据过期时回 AMO；不自动重启视觉。正常关闭 RGB 或 Ctrl+C 则整体退出。

## 本机仿真

使用 OpenCV，确保已有视频环境和模型包：

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYAV_LOGGING=off \
XDG_CACHE_HOME="$PWD/.local/cache" TORCH_HOME="$PWD/.local/cache/torch" \
HF_HOME="$PWD/.local/cache/huggingface" \
YOLO_CONFIG_DIR="$PWD/.local/video_runtime" MPLCONFIGDIR="$PWD/.local/video_runtime" \
YOLO_AUTOINSTALL=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/run_ufo_camera_demo.py \
  --with-amo --variant tcn --camera-backend opencv --camera 0 --show-detection-box
```

仿真键盘：A/Y/B/X 与真机流程一致，空格对应 R2；方向键控制平移，J/L 控制转向。RGB 窗口 q/Esc 或终端 Ctrl+C 退出。此入口固定使用 `g1_ufo_video_camera_amo` 的 MuJoCo 环境，不实例化 DDS。键盘是全局监听，测试时请聚焦仿真窗口，避免在聊天或编辑器输入操作字母。

模拟器默认在 A、等待 Y 和首次 AMO 交接时固定根部，模拟外部吊装；交接完成后释放，后续 AMO 站立／行走由物理仿真和策略自身完成。正常 R2 时重新提供仿真支撑。`sim_initial_support=False` 可关闭此仿真夹具；它不会改变真实机器人的行为。

## 指标和自动验证

每次运行仍输出原有日志、`vision_metrics.json` 和 `control_metrics.json`。后者新增 `dual_policy`，包含全程模式计数、状态切换事件、近期样本、AMO/UFO 推理 P95 及全程最大耗时。`policy_steps_total` 是累计成功下发的 UFO 目标条数（含过渡）；`policy_steps` 仍表示保留的最多 4096 条视频记录。控制周期最大耗时和超时次数全程累计，P95 范围明确标记为最近 4096 周期。

自动验证命令（socket/DDS 测试需允许本机网络）：

```bash
PYTHONPATH="$PWD:$PWD/tests" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m pytest -q tests/test_ufo_amo.py tests/test_ufo_amo_integration.py \
  tests/test_workstation_cli.py tests/test_g1_workstation_dds.py \
  tests/test_ufo_video_source.py tests/test_ufo_video.py tests/test_ufo_video_real.py \
  tests/test_ufo_streaming.py tests/test_ufo_m0_real.py
```

DDS 回环固定 domain 121、lo、禁用组播，仅与本机测试进程交换真实 HG 消息。它不模拟动力学。MuJoCo 动力学测试单独验证支撑释放后的 60 秒零速度站立、5 秒行走及回中。相机回归另行测试真实视觉输入。上述检查均不构成真机稳定性验收。

### 2026-11-08 验证结果

- 完整回归：76 通过、1 跳过（缺少既有 LAFAN 样本），另有 2 个子测试通过；随后新增的退出顺序保护与入口回归共 9 项通过。
- AMO 释放支撑后站立 60 秒，漂移 0.0315m；5 秒行走距离约 0.643m，回中后正常。最低基座高度约 0.740m，最大倾角约 0.063rad。
- 隔离 DDS：1000 个健康跟随周期约 20.11 秒，模拟端收到 1763 条指令，CRC／字段错误均为 0。最终故障锁定来自测试有意暂停反馈 160ms；恢复后 A 无法清除该故障，符合预期。
- 用户指定 OpenCV camera 0，双窗口回归完成 A→Y→B→Y，用户确认画面及切换正常；6000 个控制循环内包含 561 次 UFO 推理、3000 次 AMO 推理，视频年龄 P95 约 296ms，未记录姿态故障。
- GUI 回归控制计算 P95 约 5.56ms，但全程有 6 个慢周期，最大约 825ms。该指标包含渲染／初始化，未记录逐周期时间明细，不能断言具体慢帧来源，也不能作为真机实时性通过的证据。视频年龄不是机器人实际物理响应时间。
- 详细汇总：`.local/setup/ufo-amo-validation.json`；图形回归原始日志在 `.local/camera_demo/run_20261108_174710_22660/`。此轮未启动真实机器人动作，未提交或推送。
