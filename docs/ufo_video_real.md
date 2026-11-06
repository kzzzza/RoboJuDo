# UFO 摄像头控制 G1：双机真机测试

Jetson 的代码同步、Python 环境和驱动安装见 [端侧部署说明](ufo_video_jetson_setup.md)。

GPU 主机运行摄像头与视觉模型，G1 Jetson 运行 RoboJuDo 50 Hz 控制。`z[256]` 经 SSH 隧道传输；Wi-Fi 或网线均可承载 SSH。两台机器使用同一代码版本，模型权重留在各自本地，不提交 Git。Jetson 只需 `assets/models/g1/ufo_m0/` 中的 Actor ONNX、元数据和哈希清单；GPU 主机需要 `.local/rgb_bfm/` 三套视觉模型包。按 [Unitree 安装说明](unitree_setup.md)在 Jetson 安装 `unitree_cpp` 和官方 `unitree_sdk2py`。

## 启动

1. **吊装 G1，使其处于零力矩状态。** 确认 Jetson 的机器人控制网卡为配置中的 `eth0`，GPU 主机可以用 SSH 密钥免交互登录 Jetson。此顺序遵循 [Unitree 真机部署指南](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/README.zh.md)。
2. 在 **GPU 主机**的 RoboJuDo 目录运行。先启动此命令，摄像头程序会持续计算最新 `z`；Jetson 控制进程尚未启动时，旧帧不会排队等待播放。

   ```bash
   .venv-video/bin/python scripts/run_ufo_video_source.py \
     --camera 0 --camera-backend k4a --show-camera \
     --variant tcn --packages .local/rgb_bfm --device cuda:0 \
     --ssh-target USER@JETSON_IP --tcp-port 5541
   ```

   普通 USB 摄像头改用 `--camera-backend opencv`。需要人物框时加 `--show-detection-box`。`--variant` 也可选 `original` 或 `mlp`，两端必须相同。三模式共用流式 TCAM；Original 额外等待约 32 个逻辑帧做接触后处理与 B 编码。视觉进程处理落后时只保留最新图像，不排队播放旧动作。
3. 在 **Jetson** 的 RoboJuDo 目录先运行只读检查，再启动统一 Pipeline：

   ```bash
   .venv/bin/python scripts/check_ufo_m0_real.py \
     --config g1_ufo_video_real_stream --latent-variant tcn
   UFO_REAL_ROBOT_OK=1 .venv/bin/python scripts/run_pipeline.py \
     -c g1_ufo_video_real_stream --latent-variant tcn --latent-port 5541
   ```

   启动时程序用官方 `MotionSwitcherClient` 释放自带运动服务并复查模式；失败就退出，不创建低层控制环境。视频接收仅监听 Jetson 的 `127.0.0.1:5541`。首次试验应保持吊装支撑，逐步验证站姿和动作。

## 遥控器

| 操作 | 行为 |
| --- | --- |
| 启动后 | 不下发关节目标。 |
| 短按 **A** | 从实测关节角经过 500 个控制周期（约 10 秒）渐进到默认站姿，随后持续保持站姿。 |
| 短按 **B** | 仅在已经收到有效 `z` 时开始视频策略跟随；按下时若没有有效 `z`，保持站姿，待 `z` 有效后需再短按一次。 |
| 按 **R2** | 下一控制周期优先停止策略推理，用有效实测关节角持续下发位置保持目标。松开 R2 后仍保持停止；若要重新启动，确保 A 已松开，再短按 A，完成站姿过渡后短按 B。 |
| 短按 **X** | 视频跟随期间经过约 5 秒回到默认站姿。 |

按键按下的瞬间触发切换，A、B、X 都不需要一直按住。摄像头检测不到人时，视觉进程立即发送“人物丢失”状态；Jetson 停止策略推理，经过约 5 秒回到默认站姿并保持。人物重新入画、视觉进程重新产生有效 `z` 后，**再短按一次 B** 恢复跟随。如果此前一直按着 B，需先松开再按，因为持续按住不会产生新的按下事件。通信中断或数据过期仍进入原有的故障停止状态；R2 仍优先于回站姿动作。

超过 5 秒的 `z`、通信中断、无效反馈、姿态异常或推理异常会停止视频策略；故障停止需检查原因并重新启动 Jetson 进程。R2 是**软件停止并保持位置**，不代替 G1 的硬件急停。按 `Ctrl+C` 退出时，程序尝试用新鲜实测角度保持，再关闭驱动；关闭后的支撑行为由底层驱动决定，退出前维持吊装支撑。可在主机视觉命令和 Jetson 控制命令分别加 `--metrics-json <路径>`，核对采集到发送、采集到 PD 目标的 P95。

## 验证边界

本机回环和模拟驱动测试只验证传输、状态切换与保护逻辑。5 秒是采集到端侧控制的最大数据年龄配置；流式模式的实际 P95 需要分别核对。Jetson 运行周期、SSH 链路和机器人动作效果仍须按上述流程在真机上测量。
