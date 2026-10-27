# RoboJuDo 项目说明（供后续模型部署任务使用）

## 定位与入口

- 本仓库把 Controller（输入/动作指令）、Environment（仿真或真机反馈与执行）、Policy（观测构造与模型推理）装配成 Pipeline，用于人形机器人 sim2sim 和 sim2real。主要支持 Unitree G1、H1；以实际注册的配置与实现为准，不要仅凭 README 的支持矩阵判断某模型可运行。
- 在本文件所在目录运行 `python scripts/run_pipeline.py -c <配置名>`；省略 `-c` 时使用 `g1`。ProtoMotions tracker 使用 `python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker --motion-path <运动文件>`，该入口另支持 `--onnx-path`、`--motion-index`。
- `scripts/run_pipeline.py` 通过 `ConfigManager` 从 `cfg_registry` 取得配置，再按 `cfg.pipeline_type` 创建 Pipeline；仿真直接循环 `step()`，真机先 `prepare()`。除 `run_fullspeed=True` 外，外层按策略频率对应的 `dt` 限速。配置是 Pydantic 类，主要定义在 `robojudo/config/g1/g1_cfg.py`、`g1_loco_mimic_cfg.py`、`g1_custom_cfg.py` 及对应的 `h1/` 文件；模块导入时注册到 `cfg_registry`。

## 一次控制循环的数据流

`Environment.update()` → `Environment.get_data()` → `CtrlManager.get_ctrl_data(env_data)` → `PolicyWrapper.get_observation(env_data, ctrl_data)` → `PolicyWrapper.get_pd_target(obs)` → `Environment.step(pd_target, hand_pose)` → Pipeline、Controller、Policy 的 `post_step_callback`。

- `env_data` 是 `box.Box`，由 `robojudo/environment/base_env.py` 生成，含 `dof_pos`、`dof_vel`、`base_quat`、`base_ang_vel` 等；位置、线速度、躯干姿态和 `fk_info` 是可选的，取决于环境与里程计/FK 配置。关节角单位为弧度，`base_quat` 采用 `[x,y,z,w]`。
- `ctrl_data` 由 `robojudo/controller/ctrl_manager.py` 聚合，结构为 `{控制器类型名: 该控制器数据, "COMMANDS": [触发命令...]}`。控制器按配置列表顺序执行；`ControllerHook` 还能读取先前控制器的数据和 `env_data`。同一 `ctrl_type` 重复配置会被跳过。键盘、手柄、机器人遥控器、运动库及 Redis 都通过此接口进入策略；策略类负责解释各自需要的字段。
- `Policy.get_observation()` 构造模型输入并返回 `(obs, extras)`；`get_action()` 推理；`PolicyWrapper` 将动作加上策略默认关节姿态，形成绝对关节 PD 目标。`extras` 可携带 `hand_pose` 和 `CALLBACK`（例如 `[MOTION_DONE]`）。Pipeline 消费 `ctrl_data["COMMANDS"]`，处理关机、仿真重生、运动复位/淡入淡出及策略切换，然后将命令传给控制器和策略回调。
- `PolicyWrapper` 用 `DoFAdapter` 按**关节名称**把环境位置/速度重排到策略观测顺序，再把策略动作重排到环境顺序；环境未被策略控制的关节以环境默认姿态填充。`Environment.update_dof_cfg(override_cfg=policy.cfg_action_dof)` 将策略的 PD 增益等按关节名合并到环境配置。新增模型时必须核对 `obs_dof`、`action_dof`、环境关节名、输出维度和默认姿态。
- 常规 `RlPipeline` 在初始化后进行 10 次 `dry_run` 自检：会读取控制器并执行策略推理，但不调用 `env.step`。`RlMultiPolicyPipeline` 用 `PolicyManager` 做延迟切换；`RlLocoMimicPipeline` 在 locomotion 与 mimic 策略间做关节目标插值。切换时会复位环境并更新对应的关节 PD 配置。

## 模型如何加载

- 策略配置 `PolicyCfg.policy_type` 选择注册的策略类，`policy_file` 解析模型路径；资产根目录由 `robojudo/config/global_path.py` 指向 `assets/`。策略代码位于 `robojudo/policy/`，机器人专用配置位于 `robojudo/config/{g1,h1}/policy/`。运行时由 `PolicyWrapper` 根据 `policy_type` 实例化策略。
- `disable_autoload=False` 的策略在 `Policy.__init__` 中调用 `torch.jit.load(policy_file, map_location=device)`；默认 `get_action()` 使用 Torch 推理，并按 `action_beta`、`action_clip`、`action_scale` 处理输出。Unitree、Smooth、H2H、AMO、TWIST 走此路径；AMO 还单独加载 adapter TorchScript 和归一化统计文件。不能把任意 `.pt` checkpoint 直接当作 TorchScript 使用。
- `disable_autoload=True` 的策略自行加载 ONNX：BeyondMimic、ASAP mimic/locomotion、KungfuBot2 和 ProtoMotions tracker 使用 `onnxruntime.InferenceSession`，各类自行构造输入/处理输出。BeyondMimic 可从 ONNX model metadata 读取关节名、默认姿态、增益及动作缩放；ProtoMotions 从同名 `.yaml` 读取关节/控制元数据，并另加载 `motion_path` 指定的运动文件，模型输出为绝对 PD 目标。新增 ONNX 模型须确认元数据、输入输出名称/形状及对应策略类的预处理约定。
- 路径约定见 `robojudo/policy/policy_cfgs.py`：Unitree/Smooth/H2H/TWIST 通常在 `assets/models/<robot>/<family>/<policy_name>.pt`；BeyondMimic 在 `.../beyondmimic/<policy_name>.onnx`；ASAP mimic 在 `.../asap/mimic/<policy_name>/<relative_path>`，locomotion 在 `.../asap/dec_loco/<policy_name>/<relative_path>`；KungfuBot2 在 `.../kungfubot2/<policy_name>.onnx`。ProtoMotions 路径由专用配置的 `onnx_name` 或 `onnx_path` 给出。
- `assets/models/` 下存在部分预置文件，但是否能运行要逐一核对配置引用、运动文件、可选子模块及依赖。HugWBC/GMT 虽有权重，当前 `robojudo/policy/__init__.py` 未注册相应策略类，不能仅凭权重文件直接运行。

## 模块间及外部通信

- Pipeline 内部主要是同步 Python 对象/`Box` 传值，不依赖消息总线。MuJoCo 环境读取 `qpos/qvel`，按策略给出的 PD 目标计算 `(target-pos)*kp - vel*kd`，每个策略周期执行 `sim_decimation` 个 MuJoCo 步；仿真 XML 路径由环境配置选择。
- 真机环境是 `UnitreeEnv`（`unitree_sdk2py`，DDS 的 lowstate/sportstate 订阅及 lowcmd 发布）或 `UnitreeCppEnv`（`unitree_cpp.UnitreeController` 的状态读取与 `step` 命令）。`UnitreeCtrl` 借环境的 `RemoteControllerHandler` 接收无线遥控数据。`net_if`、消息类型、`joint2motor_idx`、里程计、控制周期和手部设置都在真机环境配置中。
- `g1_twist` 示例默认使用 `TwistRedisCtrl`：后台线程从配置的 Redis key（G1 默认 `action_mimic_g1`）读取 JSON 数组，作为 `ctrl_data["TwistRedisCtrl"]["action_mimic"]` 交给 `TwistPolicy`；也可在配置中换成内置 `MotionTwistCtrl`。运动控制器通常从 `assets/motions/` 加载轨迹，再以控制器数据传给对应策略。

## 后续添加/测试模型的操作顺序

1. 先确定机器人、环境（MuJoCo/真机）、模型格式、控制频率、关节顺序、观测结构、输出语义和所需动作/运动源。阅读目标策略实现和对应 `*PolicyCfg`，不要只改模型文件名。
2. 将模型及需要的元数据/轨迹放到 `assets/models/`、`assets/motions/` 的约定位置，或给配置提供明确路径。必要时新建 `Policy` 子类与策略配置，在 `robojudo/policy/__init__.py` 注册；新控制器/环境也需在相应包的 `__init__.py` 注册。
3. 在 `robojudo/config/g1/` 或 `h1/` 定义带 `@cfg_registry.register` 的完整 Pipeline 配置，并确保该模块被对应包 `__init__.py` 导入。选择匹配的 `env`、`ctrl`、`policy`/`policies`。`robojudo/config/config_factory.py` 标为 TODO，不是当前 CLI 的主要配置路径。
4. 先做配置、文件存在性和输入输出维度检查，再在 MuJoCo 跑 sim2sim；重点检查关节名映射、默认姿态、PD 增益、动作缩放、频率及命令来源。`tests/test_full_imports.py` 只验证已注册模块能否导入，不证明模型能安全运行。
5. 真机配置（如 `g1_real`、`h1_real`）需单独核对 SDK、网卡 `net_if`、电机索引和里程计；确认急停映射与 `do_safety_check`，完成仿真验证后再运行。仓库默认配置的真机运行会在入口中调用 `prepare()`，使关节渐进到初始姿态；不要绕过此流程。

关键参考：`README.md`、`docs/policy.md`、`docs/controller.md`、`docs/environment.md`、`docs/unitree_setup.md`。当文档与代码不一致时，以当前代码和所选配置为准。

## UFO M0 G1 sim2sim

- 原版 M0 使用同包的 B 与 Actor 两个 ONNX；注册配置 `g1_ufo_m0`，有限时长验证入口 `scripts/run_ufo_m0.py`，用法和数据约定见 `docs/ufo_m0.md`。
- `~/data/lafan/lafan_g1_npz` 的 G1 NPZ 关节及刚体数组顺序均与 UFO XML 不同，四元数为 `wxyz`；转换逻辑在 `robojudo/utils/ufo_m0_motion.py`。更换动作来源时先确认这些约定。
- `assets/robots/g1/g1_ufo_m0.xml` 在 UFO 机器人 XML 上补充地面、电机转子惯量及干摩擦；直接复制原 XML 会使仿真迅速发散。
- MuJoCo 参考虚影和 EGL 无头视频是 `MujocoEnv` 的通用功能；策略在 `extras["reference_qpos"]` 传入下一控制帧完整的模型 `qpos`，用法见 `docs/mujoco_rendering.md`。UFO 默认使用浅色背景；ONNX 模型文件仅保留本地，不纳入 Git。
- UFO M0 真机配置为 `g1_ufo_m0_real`，仍通过 `scripts/run_pipeline.py` 运行并沿用 RoboJuDo 的 `UnitreeCppEnv`。启动前运行只读 `scripts/check_ufo_m0_real.py`，并设置 `UFO_REAL_ROBOT_OK=1`。其 `prepare()` 有意不自动下发，需遥控器 A 初始化站姿、R1 允许策略、B 开始跟踪；R2 锁定停止，完成站姿初始化后可在释放 R1/B 后用 R1+B 重新启动，未完成站姿则需重按 A。故障停止需要检查后重启进程。详见 `docs/ufo_m0.md`。
- UFO M0 Actor 的四帧历史按新到旧排列，先读历史再加入当前帧；本地 Actor/B 文件由 `assets/models/g1/ufo_m0/artifact_manifest.json` 记录哈希，不提交 ONNX。Real 配置单独启用关节位置及逐帧变化保护，其他策略默认不启用。
