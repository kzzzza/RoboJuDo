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

脚本依次启动原始、TCN、MLP 的视觉进程和 MuJoCo 控制进程。总览见 `$EXP_DIR/stream/summary.json`；各组的 `worker.json`、`control.json` 和日志位于同名子目录。流式算法使用滚动窗口，其 `z` 与步骤 1 的完整离线结果不能视为相同算法输出。

## 可选：仅诊断潜变量传输与控制延时

```bash
.venv-video/bin/python scripts/run_ufo_video_experiments.py \
  --mode transport --offline "$EXP_DIR" --output "$EXP_DIR/transport"
```

此命令按时间戳发送步骤 1 **预计算**的 `z`，结果在 `$EXP_DIR/transport/summary.json`；它不运行视频推理，不能用于判断完整视觉链路是否达到实时要求。

## 摄像头双窗口演示

接好 Azure Kinect DK 后，站在镜头前并保持全身入画。在桌面终端运行：

```bash
.venv/bin/python scripts/run_ufo_camera_demo.py --variant tcn
```

脚本同时打开 `Camera RGB` 和 MuJoCo 窗口。摄像头画面经视觉模型生成 UFO 的 `z`，再由同一个 `LatentCtrl` 和 `UfoM0Policy` 控制仿真机器人。按摄像头窗口中的 `q` 或 `Esc`，或者在终端按 `Ctrl+C` 退出。可将 `tcn` 改为 `mlp` 或 `original`；每次运行的日志与指标保存在 `.local/camera_demo/run_*/`。如果使用普通 UVC 摄像头，添加 `--camera-backend opencv --camera 0`。

需要在 RGB 窗口查看人物检测框时，添加 `--show-detection-box`；默认不绘制。框来自当前控制链路使用的人物检测结果。

此演示配置允许视觉推理延迟较大，用于观察模型动作效果；真机视频配置仍关闭。
