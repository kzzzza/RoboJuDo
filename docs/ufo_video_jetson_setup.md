# G1 Jetson 端侧部署：从本机同步 RoboJuDo

本文只安装 **Jetson 上的控制端**。GPU 主机运行摄像头和视觉模型，Jetson 运行 `g1_ufo_video_real_stream`，通过 SSH 隧道接收 `z[256]`。两端必须使用同一次同步对应的代码协议。以下命令中的 `USER@JETSON_IP` 换成实际 SSH 地址。

## 1. 在本机同步代码与 Actor

在**本机 GPU 主机**执行。同步当前工作目录的代码，包含尚未提交的修改；不会把 x86 虚拟环境、视觉模型、实验输出或 Git 历史复制到 Jetson。再次部署时重复这一步即可，命令不删除 Jetson 上已有文件。

```bash
cd /home/amax/workspace/deploy/RoboJuDo
git status --short
JETSON=USER@JETSON_IP
ssh -o BatchMode=yes "$JETSON" 'true'
ssh "$JETSON" 'mkdir -p ~/RoboJuDo/assets/models/g1/ufo_m0'
rsync -av --info=progress2 \
  --exclude='.git' --exclude='.venv/' --exclude='.venv-video/' \
  --exclude='.local/' --exclude='logs/' --exclude='recordings/' \
  --exclude='__pycache__/' --exclude='*.egg-info/' \
  --exclude='/assets/models/' --exclude='/assets/motions/' \
  ./ "$JETSON:~/RoboJuDo/"
rsync -av --info=progress2 \
  assets/models/g1/ufo_m0/FBcprAuxModel.onnx \
  assets/models/g1/ufo_m0/FBcprAuxModel.meta.json \
  assets/models/g1/ufo_m0/artifact_manifest.json \
  "$JETSON:~/RoboJuDo/assets/models/g1/ufo_m0/"
```

`FBcprAuxModel.onnx` 约 128 MB，仅作为部署文件传输，仍由 Git 忽略。端侧视频控制**不需要** `backward_encoder.onnx`、LAFAN 动作、`.local/rgb_bfm` 或摄像头驱动。若本机的 `packages/unitree_cpp/` 为空，而端侧尚未安装该驱动，先在本机运行 `git submodule update --init packages/unitree_cpp`，然后重跑上述 `rsync`；不要将 x86 编译产物当作 Jetson 驱动使用。

## 2. 在 Jetson 创建 ARM 环境

在 **Jetson** 执行。项目要求 Python 3.11 或更新版本；先核对 JetPack、架构和解释器。虚拟环境必须在 Jetson 本机创建，不能复制本机的 `.venv`。

```bash
ssh USER@JETSON_IP
cd ~/RoboJuDo
uname -m
test -f /etc/nv_tegra_release && cat /etc/nv_tegra_release
python3.11 --version
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -c 'import torch, onnxruntime, robojudo; print("Python dependencies OK")'
```

预期架构为 `aarch64`。若系统没有 Python 3.11，须先安装与该 JetPack 兼容的 Python 3.11，再创建虚拟环境。`torch` 在 RoboJuDo 包初始化时需要导入，Actor 实际使用 ONNX Runtime 的 **CPUExecutionProvider**。若通用 `pip` 无法为端侧 Python/ARM 找到兼容的 `torch` 或 `onnxruntime`，按端侧 JetPack/Python 版本安装对应 ARM wheel 后重试；不要使用本机 x86 wheel。PyTorch 的 JetPack wheel 选择见 [NVIDIA 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/)，ONNX Runtime 安装方式见 [官方文档](https://onnxruntime.ai/docs/install/)。

## 3. 安装 G1 驱动

如果这台 Jetson 已装好并验证过 RoboJuDo 的 `unitree_cpp` 驱动，可保留现有驱动，但要在**上述 `.venv` 中**确认两个 Python 包均可导入：

```bash
cd ~/RoboJuDo
.venv/bin/python -c 'import unitree_cpp, unitree_sdk2py; print("G1 SDK imports OK")'
```

全新安装时，先在 Jetson 安装真实的 C++ SDK，再从同步来的 `packages/unitree_cpp/` 在 Jetson 本机编译安装绑定，最后在同一 `.venv` 安装官方 `unitree_sdk2py`。以下命令需要 Jetson 能访问软件源和 GitHub；若端侧离线，先在本机下载这两个 SDK 的**源码**并用 `rsync` 传到 Jetson，仍须在 Jetson 上编译：

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libfmt-dev
git clone https://github.com/unitreerobotics/unitree_sdk2.git ~/unitree_sdk2
cmake -S ~/unitree_sdk2 -B ~/unitree_sdk2/build
cmake --build ~/unitree_sdk2/build -j2
sudo cmake --install ~/unitree_sdk2/build
cd ~/RoboJuDo
test -f packages/unitree_cpp/CMakeLists.txt
.venv/bin/python -m pip install ./packages/unitree_cpp
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ~/unitree_sdk2_python
.venv/bin/python -m pip install -e ~/unitree_sdk2_python
.venv/bin/python -c 'import unitree_cpp, unitree_sdk2py; print("G1 SDK imports OK")'
```

`unitree_cpp` 是 50 Hz 关节控制驱动；`unitree_sdk2py` 用于启动时释放 G1 自带运动服务、进入 debug 模式。不要使用 `unitree_cpp` 的 dummy SDK 构建。若 `unitree_sdk2py` 安装报 CycloneDDS 错误，按 [官方安装说明](https://github.com/unitreerobotics/unitree_sdk2_python#installation)先配置 CycloneDDS；C++ SDK 的安装路径与依赖见 [官方说明](https://github.com/unitreerobotics/unitree_sdk2/blob/main/README.md#installation)。复制代码的目录不含 `.git`，因此端侧不要运行依赖 Git 子模块元数据的 `submodule_install.py`。

## 4. 只读预检与启动

先检查 Jetson 上连接 G1 的网卡名称及状态。当前配置使用 `eth0`；若实际名称不同，修改 `robojudo/config/g1/env/g1_real_env_cfg.py` 中对应 `net_if` 配置并重新同步。预检只检查模型、配置、依赖和网卡，**不创建控制环境、不下发动作**。

```bash
cd ~/RoboJuDo
ip -br link
.venv/bin/python scripts/check_ufo_m0_real.py \
  --config g1_ufo_video_real_stream --latent-variant tcn
```

检查通过后，按 [双机真机操作流程](ufo_video_real.md)先启动 GPU 主机视觉进程，再在吊装支撑的 G1 Jetson 上启动：

```bash
cd ~/RoboJuDo
UFO_REAL_ROBOT_OK=1 .venv/bin/python scripts/run_pipeline.py \
  -c g1_ufo_video_real_stream --latent-variant tcn --latent-port 5541
```

视觉端与控制端的 `--variant` / `--latent-variant` 必须一致。控制端只监听 Jetson 的 `127.0.0.1:5541`，由主机启动脚本建立 SSH 隧道。启动会自动尝试切换 debug 模式；A、B、X、R2 的操作及真机验证边界见 [双机真机操作流程](ufo_video_real.md)。

## 5. 更新代码

本机修改后重新执行第 1 节的两条 `rsync`，在 Jetson 停止旧 Pipeline 后重新运行第 4 节预检与启动。若 `requirements.txt` 或驱动源码变更，在 Jetson 重新执行相应安装命令。保留 Jetson 的 `.venv`，不要从本机覆盖它。
