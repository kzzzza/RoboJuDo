# UFO 实时视觉低延迟部署

摄像头的 CUDA TCN／MLP 默认选择 `--vision-profile auto`：每个处理完成的新图像产生一个最新 z，HMR2、TCAM 特征与 TCN 的 14 帧历史留在 GPU。Original、CPU 和默认文件回放继续使用旧流式算法；完整离线生成算法不变。`--vision-profile legacy` 可显式回退。

本实现参考 `robojudo_ly/scripts/run_rgb2sm_vision.py` 的 TensorRT HMR2、GPU 特征缓存和输出节奏。保留本项目的 YOLOv8x、HMR2、TCAM 和 student 权重，检测每帧执行。参考实现使用不同的 YOLO 权重并隔帧检测，其速度数字不能直接当作本链路的测量值。摄像头检测不到人时立即清除窗口与历史并发送 `person_lost`，不继续使用旧检测框生成 z。

## 安装与构建

使用项目已有 `.venv-video`；下面的脚本拒绝在项目以外的环境安装。TensorRT 依赖较大，首次下载和引擎构建需要时间。

```bash
bash scripts/setup_ufo_vision_accel.sh
PYTHONPATH="$PWD" XDG_CACHE_HOME="$PWD/.local/cache" \
  .venv-video/bin/python scripts/prepare_ufo_vision_accel.py
```

固定依赖为 TensorRT CUDA12 `10.13.3.9`、ONNX `1.19.1`，并保持 NumPy `1.26.4`／ml_dtypes `0.5.3`，避免间接依赖升级破坏已有 Ultralytics 环境。缓存全部位于 `.local/vision_accel`，包含 TCAM 的 FP16 TorchScript、HMR2 与 YOLOv8x 的 TensorRT 引擎及验证清单。源权重不改写。YOLO 引擎允许 batch=1、256～640 的动态输入高宽，沿用 Ultralytics 的 letterbox 与 NMS；HMR2 输入固定为 `[1,3,256,256]`。

检测器使用 FP16 卷积，动态 Range／坐标解码保留所需的 FP32，以兼容 TensorRT 的算子约束。精度选择使用 [TensorRT 10 的精度控制接口](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)。

构建脚本验证 TCAM 的不同过去上下文掩码、HMR2 数值，以及检测器多种输入尺寸，全部通过才写入清单。运行时校验源权重／实现哈希、产物哈希、Torch、GPU 和对应运行库版本。缓存不匹配会要求重建。没有缓存时，`auto` 使用同权重的 PyTorch FP16；显式指定 TensorRT 时缺少有效引擎会报错。

仅准备 TorchScript、不安装 TensorRT时，可运行：

```bash
PYTHONPATH="$PWD" .venv-video/bin/python scripts/prepare_ufo_vision_accel.py \
  --hmr-backend torch --detector-backend torch
```

## 本机摄像头与 MuJoCo

```bash
.venv/bin/python scripts/run_ufo_camera_demo.py \
  --variant tcn --with-amo --camera-backend opencv --camera 0 \
  --show-detection-box
```

两个后端均默认 `auto`，有有效缓存则使用 TensorRT。可添加 `--hmr-backend torch --detector-backend torch` 测量纯 Torch FP16，或者 `--vision-profile legacy` 对比旧版。AMO／UFO 的 A→Y→B、Y/X 返回 AMO、空格模拟 R2 的流程不变。

摄像头原始图像的方向与显示器旋转设置无关。如果原始画面横转，可使用 `--camera-rotate 270`（顺时针 270°，即逆时针 90°）。默认 0；旋转在检测和裁剪前完成。该选项同样适用于 Azure Kinect。

工作站入口 `scripts/run_ufo_workstation_real.py` 接受相同的视觉参数，继续使用原有 DDS、反馈保护、目标限幅和使能条件。参考 [工作站 DDS 部署](ufo_video_workstation.md) 和 [AMO 双策略流程](ufo_video_amo.md)。本次视觉优化只做本机验证，不能据此认定真机动作稳定性已经验收。

统一入口在机器人控制进程启动前检查视觉缓存并加载、预热所选后端。也可独立运行只检查模型、不开相机／socket 的命令：

```bash
.venv-video/bin/python scripts/run_ufo_video_source.py \
  --camera 0 --variant tcn --check-runtime
```

## 同一片段复测

以下命令仅启动视觉与本机 Unix socket 接收器，没有机器人或 DDS 控制进程。以摄像头语义按视频时间戳重放，保留最新一帧，检测丢失会复位，统计有效新 z 的实际发送频率。每次使用新的结果目录。

```bash
PYTHONPATH="$PWD" .venv-video/bin/python scripts/benchmark_ufo_vision.py \
  --video /absolute/path/person.mp4 --vision-profile legacy \
  --output-dir .local/vision_benchmark/baseline
PYTHONPATH="$PWD" .venv-video/bin/python scripts/benchmark_ufo_vision.py \
  --video /absolute/path/person.mp4 --vision-profile low-latency \
  --output-dir .local/vision_benchmark/accelerated
PYTHONPATH="$PWD" .venv-video/bin/python scripts/validate_ufo_vision_accel.py \
  --video /absolute/path/person.mp4 --hmr-backend tensorrt \
  --output .local/vision_benchmark/numerical_agreement.json
```

数值验证对相同的 50 张真人图像裁剪，逐级比较 FP32 与加速后的 HMR2 特征、TCAM u 和 TCN z。验证范围包括 0/1/3/8 个过去上下文，未来槽始终不启用，并检查 z 的维度、有限值及范数 16。它验证运行精度变化；新旧流式调度本身的输出不要求逐位相同。

## 指标定义

- `output_rate_hz`：实际发送的有效新 z 频率。控制仍以 50Hz 读取最新 z；两者分别统计，不靠重复发送旧值增加频率。
- `update_interval_p50_s/p95_s`：连续有效新 z 的发送间隔。人物重新入画的预热间隔也会计入。
- `capture_to_send_p50_s/p95_s`：源时间戳到本机 z 发送完成，包含取到最新图像时的年龄、预处理与模型计算。视频复测的源时间戳由回放 PTS 产生；摄像头时间戳从主机读取到帧时开始。均不包含此前的曝光／传感器缓存、机器人 DDS 接收及机械响应。
- `detector_*`、`preprocess_*` 是同步完成的主机耗时，`hmr_*`、`tcam_*`、`backend_*` 用 CUDA event 在最终同步后统计，避免把 GPU 排队时间误写成推理时间。
- `hmr_backend`、`detector_backend`、`temporal_backend` 记录实际后端；日志中的 30Hz 是逻辑采样时间轴，不是声称模型每秒处理 30 张新图像。

TCAM 仍使用 50 个逻辑样本，初始等待约 1.67 秒。上下文均值每 4 个逻辑样本更新，与新 z 的输出节奏分开；只填过去 8 个槽。相机帧被跳过时用最后已知特征补齐时间轴，并保留其真实采集时间。TCN 消费全部新逻辑 token，保留独立 14 帧历史；不会因为一次不足 4 帧的更新而重新清空历史。人物丢失或长时间缺帧会清空历史并重新预热。

## 本机验证记录（2026-11-08）

RTX 4060 Laptop 上，同一段 OpenCV 真人视频（957 帧、30FPS，转正后 480×640）按实时 PTS 重放。以下均为取帧时间戳到本机发送 z，不包含机器人网络／执行时间。

| 路径 | 有效新 z 频率 | 采集到发送中位数 | 采集到发送 P95 | 新 z 间隔中位数 |
|---|---:|---:|---:|---:|
| 原流式 FP32／4 帧更新 | 5.56Hz | 116.36ms | 132.57ms | 183.21ms |
| 逐帧／GPU 缓存／Torch FP16 | 22.63Hz | 60.44ms | 75.58ms | 44.02ms |
| 逐帧／GPU 缓存／TensorRT | 30.00Hz | 28.54ms | 29.71ms | 33.33ms |

最终运行处理全部 957 张图像，发送 908 个有效新 z，初始窗口等待 1.666 秒；取帧丢弃、人物丢失和 socket 发送失败均为 0。YOLOv8x、裁剪、HMR2、TCAM、TCN 的耗时中位数约为 9.07、1.11、9.38、6.84、1.14ms；各阶段分位数不能直接相加当作端到端分位数。

50 张相同真人裁剪的 FP32 对比中，最终 z 的最小余弦相似度约 0.999981；检测框最小 IoU 为 0.995744，平均为 0.999336。所有 z 的维度、有限值和范数检查通过。另对 TensorRT 检测器反复缩小／放大输入运行 GPU 回归，覆盖 Ultralytics 8.3 动态输出扩容后地址失效的问题；修复放在项目内适配层，不修改安装包文件。

录制的摄像头画面经过**完整实时视觉推理**、socket、两个 Actor 和 MuJoCo，完成 A/Y/B、Y 返回、R2 和重新 A/Y/B，共 2091 个控制周期、675 次 UFO 目标计算，未触发锁定故障。新 z 首次被控制使用时，采集到 Actor 关节目标生成的中位数／P95 为 41.51／47.89ms；包含重复使用上一 z 的所有 UFO 周期为 47.60／61.90ms。该时间戳在统一目标保护与 `env.step` 前采集。控制计算 P95 为 5.46ms、最大 9.90ms。这是本机软件链路测量，不是真机 DDS 或机械动作延迟。

AMO 额外完成 60 秒零速度站立、5 秒行走和回中回归；核心视觉／状态机／工作站入口测试通过。缺少 LAFAN 文件的测试按条件跳过，可选 CUDA 测试在视觉环境单独执行通过。本轮没有初始化机器人 DDS 或运行真机动作。

本机详细产物位于 `.local/vision_benchmark/`，包含各路径 `metrics.json`、`tensorrt_parity.json`、`simulation/report.json` 和 `amo_60s_regression.json`。日志保存在 `.local/setup/vision-*.log`；个人测试视频与缓存不提交 Git。
