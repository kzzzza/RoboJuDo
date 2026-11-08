# UFO M0 视频控制：本机复现实验

在 RoboJuDo 仓库根目录执行以下命令。控制环境为 `.venv`，视觉环境为 `.venv-video`；确认三套模型包位于 `.local/rgb_bfm/{original,tcn,mlp}_pipeline/`。模型权重和实验结果均留在 Git 忽略的 `.local/`。每次复现实验使用一个**新的**输出目录；离线准备脚本不会覆盖已有结果。

```bash
VIDEO_PATH="$HOME/Videos/dance_bilbil.mp4"
EXP_DIR=.local/video_runs/manual_repro
```

## 1. 生成三组离线潜变量

```bash
.venv-video/bin/python scripts/prepare_ufo_video_experiment.py \
  --video "$VIDEO_PATH" --output "$EXP_DIR" --device cuda:0
```

原始版从视频生成 `u` 和 `z`；TCN、MLP 使用同一份 `u` 分别生成 `z`。结果位于 `$EXP_DIR/original/`、`tcn/`、`mlp/`，对比指标在 `$EXP_DIR/offline_report.json`。

## 2. 分别运行 MuJoCo 离线闭环

```bash
for variant in original tcn mlp; do
  .venv/bin/python scripts/run_pipeline.py -c g1_ufo_video_file \
    --latent-dir "$EXP_DIR/$variant" --latent-variant "$variant" \
    --metrics-json "$EXP_DIR/${variant}_control.json" \
    --record-video "$EXP_DIR/${variant}_simulation.mp4"
done
```

三组都从相同的默认站姿开始。每组生成控制指标 JSON 和无头录制的 MP4；已有 MP4 不会被覆盖。

## 3. 按视频时间戳回放，测试完整流式链路

```bash
.venv-video/bin/python scripts/run_ufo_video_experiments.py \
  --video "$VIDEO_PATH" --offline "$EXP_DIR" \
  --output "$EXP_DIR/stream" --device cuda:0
```

脚本依次启动原始、TCN、MLP 的视觉进程和 MuJoCo 控制进程。总览见 `$EXP_DIR/stream/summary.json`；各组的 `worker.json`、`control.json` 和日志位于同名子目录。20 FPS 等非 30 FPS 输入也会按源时间戳因果采样到 30 Hz。视觉进程使用 50 帧窗口、4 帧更新；Original 额外保留 32 帧右侧等待，并每累计至少 12 个可提交帧运行一次后处理。长视频推理落后时会跳过旧图像，因此流式 `z` 不会与完整离线输出逐帧相同；报告按逻辑帧号给出对齐误差。

## 可选：仅诊断潜变量传输与控制延时

```bash
.venv-video/bin/python scripts/run_ufo_video_experiments.py \
  --mode transport --offline "$EXP_DIR" --output "$EXP_DIR/transport"
```

此命令按时间戳发送步骤 1 **预计算**的 `z`，结果在 `$EXP_DIR/transport/summary.json`；它不运行视频推理，不能用于判断完整视觉链路是否达到实时要求。

## 摄像头双窗口演示

Azure Kinect 的运行库安装与 USB 权限配置见 [Azure Kinect 本机安装与测试](azure_kinect_setup.md)。

CUDA 摄像头的 TCN／MLP 默认使用逐帧输出和 GPU 缓存的低延迟路径。TensorRT 安装、缓存构建、后端切换、相机旋转及同片段延迟复测见 [实时视觉低延迟部署](ufo_vision_latency.md)。文件回放默认保持原算法。

接好 Azure Kinect DK 后，站在镜头前并保持全身入画。在桌面终端运行：

```bash
.venv/bin/python scripts/run_ufo_camera_demo.py --variant tcn
```

脚本同时打开 `Camera RGB` 和 MuJoCo 窗口。摄像头画面经视觉模型生成 UFO 的 `z`，再由同一个 `LatentCtrl` 和 `UfoM0Policy` 控制仿真机器人。按摄像头窗口中的 `q` 或 `Esc`，或者在终端按 `Ctrl+C` 退出。可将 `tcn` 改为 `mlp` 或 `original`；每次运行的日志与指标保存在 `.local/camera_demo/run_*/`。如果使用普通 UVC 摄像头，添加 `--camera-backend opencv --camera 0`。

摄像头检测不到人时，视觉进程立即通知 MuJoCo 停止使用旧 `z`，机器人用约 5 秒回到默认站姿并保持。人物重新入画且视觉窗口重新产生有效 `z` 后，仿真自动恢复跟随。通信中断或数据过期仍按原有故障处理，不当作人物离开。

需要在 RGB 窗口查看人物检测框时，添加 `--show-detection-box`；默认不绘制。框来自当前控制链路使用的人物检测结果。

摄像头演示和真机视频现在都使用 5 秒数据年龄限制。`vision_metrics.json` 记录首次输出等待、取帧丢弃、各阶段耗时和采集到发送的延时；仿真 `control_metrics.json` 记录采集到 PD 目标的延时。超过 5 秒时停止策略输出。

G1 双机摄像头真机测试命令及遥控器操作见 [UFO 视频真机部署](ufo_video_real.md)。
