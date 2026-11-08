# Azure Kinect 本机安装与测试

Ubuntu 24.04 x86_64 工作站使用微软官方 Azure Kinect Sensor SDK 1.4.2 二进制包。官方发行包针对较早的 Ubuntu，是否可用以本机加载、设备打开和连续取帧结果为准。安装源码、包、运行库和验证结果均保存在项目 `.local/k4a/`。

## 安装

```bash
bash scripts/setup_azure_kinect.sh
```

该脚本校验固定版本运行包及开发包的 SHA256，并解压至 `.local/k4a/runtime/`，其中包含 `libk4a.so` 和配套深度引擎。它不修改系统 apt 源或 Conda 环境。项目的 `AzureKinectRGB` 优先使用系统 libk4a，找不到时自动加载此本地副本；无需设置 `LD_LIBRARY_PATH`。

普通用户访问 Kinect 需要微软官方 USB 权限规则：

```bash
sudo install -m 0644 .local/k4a/99-k4a.rules /etc/udev/rules.d/99-k4a.rules
sudo udevadm control --reload-rules
```

然后重新插拔 Kinect USB 线。也可以使用 `pkexec` 在桌面系统认证窗口中授权安装规则。连接外部电源和 USB 3 接口，`lsusb` 应能看到微软的 `045e:097c` 深度相机与 `045e:097d` RGB 相机。项目当前配置同时启用 720p/30Hz BGRA RGB 和 NFOV_UNBINNED 深度流，所以配套深度引擎及 GPU/OpenGL 依赖也必须可用。

## 仅本机仿真

打开摄像头与 MuJoCo 窗口，使用 TCN；请保持全身入画：

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 PYAV_LOGGING=off \
XDG_CACHE_HOME="$PWD/.local/cache" TORCH_HOME="$PWD/.local/cache/torch" \
HF_HOME="$PWD/.local/cache/huggingface" \
MPLCONFIGDIR="$PWD/.local/video_runtime" YOLO_CONFIG_DIR="$PWD/.local/video_runtime" \
YOLO_AUTOINSTALL=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/run_ufo_camera_demo.py \
  --variant tcn --camera-backend k4a --camera 0 --device cuda:0 \
  --show-detection-box --max-steps 2000
```

此入口只使用 `g1_ufo_video_camera` 的 MuJoCo 环境，通过本机 Unix socket 传输潜变量，不创建机器人 DDS 驱动。关闭 RGB 窗口或按 Ctrl+C 结束。日志和控制/视觉指标保存于 `.local/camera_demo/run_*/`。

若出现 `LIBUSB_ERROR_ACCESS`，检查上述 udev 规则和 USB 节点权限；`found 0 device(s)` 时检查供电、USB 线与端口；启动相机失败时先确认没有其他程序占用相机。不要通过启动真机控制入口诊断摄像头。

## 本机验证记录（2026-11-08）

- Ubuntu 24.04.3 x86_64；官方 runtime/dev 包 1.4.2 的 SHA256 校验通过，原生依赖无缺失，USB 规则已安装至 `/etc/udev/rules.d/99-k4a.rules`。
- 连续 RGB 取帧：150 帧、1280×720，实测 30.09Hz；图像非空且帧间变化正常。未保存图像内容。
- Azure Kinect → TCN → MuJoCo：2000 个循环、1121 次策略推理、124 个不同潜变量帧；采集到 PD 计算后延迟 P95 为 301.8ms，未记录摔倒。
- 视觉有效发送约 5.50Hz，与原始取帧约 30Hz 和配置控制频率 50Hz 不同。最新帧队列丢弃 549 个采集帧，记录 1 次时序重置；有限步仿真退出并关闭 socket 后，视觉结束前记录 2 次发送失败。以上不是零丢帧或真机稳定性验证。
- 相机入口与视频测试：7 项通过。报告在 `.local/k4a/validation/summary.json`；仿真原始日志和指标位于 `.local/camera_demo/run_20261108_154727_12513/`。未启动真实机器人驱动或 DDS 通信。

官方参考：[SDK 使用与设备权限](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/v1.4.2/docs/usage.md)、[深度引擎说明](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/v1.4.2/docs/depthengine.md)、[USB 规则](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/v1.4.2/scripts/99-k4a.rules)。
