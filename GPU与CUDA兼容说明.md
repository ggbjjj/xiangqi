# GPU 型号、CUDA 版本与驱动兼容说明

核对日期：2026-10-29。适用于当前 HaiKu、Opus 的训练、验证和自对弈。本文按项目的 `pyproject.toml`、`uv.lock` 和官方版本资料说明兼容范围；显卡列表表示架构兼容性判断，尚不是每款卡都完成了本项目训练实测。本机当前为 CPU 环境。

## 一、当前项目用哪个 CUDA

| 项目 | 当前配置 |
|---|---|
| 项目环境默认 Python | **3.12**，见 `.python-version`；项目声明范围为 `>=3.11,<3.14` |
| PyTorch | **2.9.1**，不随系统已有版本自动升级 |
| GPU 安装方案 | **torch 2.9.1+cu128，即 CUDA 12.8 构建** |
| 锁文件中的 NVIDIA 运行库 | CUDA Runtime **12.8.90**、cuDNN **9.10.2.21**；实际安装/随 wheel 打包的方式依平台而定 |
| 主要部署平台 | Linux x86_64（AutoDL）或 Windows x64，配 NVIDIA GPU 与兼容驱动 |
| Linux 系统库 | 当前 Python 3.12 的 cu128 wheel 标签为 `manylinux_2_28_x86_64`，系统 glibc 至少 2.28；见 `uv.lock` |
| CPU 方案 | `--extra cpu`，不需要 CUDA 或 NVIDIA GPU |
| 是否需要单独安装 CUDA Toolkit / nvcc | **当前项目使用预编译 wheel，不需要**；安装 GPU 依赖并确保主机 NVIDIA 驱动兼容即可 |

当前代码使用 PyTorch 自带的卷积、线性层、SDPA 注意力、Dropout、自动混合精度和重计算，没有额外编译的 CUDA 扩展，也没有要求用户另装 FlashAttention。SDPA 根据输入及硬件选择可用后端，因此显卡不支持某个融合内核时不等于整个网络不能运行。[PyTorch 2.9 SDPA 官方说明](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 二、“最低 CUDA”要怎样理解

| 问题 | 答案 |
|---|---|
| 不改项目配置，运行 `uv sync --locked --extra gpu` 会使用哪个版本？ | **CUDA 12.8 wheel**。这是当前项目的默认支持方案。 |
| 主机必须事先装 CUDA 12.8 Toolkit 吗？ | **不必**。本项目运行时使用 PyTorch wheel 及其依赖提供的 CUDA 库。 |
| PyTorch 2.9.1 官方提供的 CUDA wheel 最低是哪一版？ | **12.6**；另提供 12.8、13.0。12.6 尚未配置为本项目的安装选项，不能把它写成当前锁定环境的默认版本。 |
| 代码理论上绝对最低能跑哪个 CUDA/PyTorch 组合？ | 尚未逐版测试，不能给出保证。本文只说明当前锁定环境及官方可选构建。 |

PyTorch 2.9.1 的可选 wheel 版本依据 [官方历史版本安装表](https://pytorch.org/get-started/previous-versions/)。需要改成 cu126 时，应同步调整 `pyproject.toml` 的 GPU 索引、重新生成 `uv.lock`，再验证安装与训练；只在环境中用 pip 替换 torch，后续 `uv run --locked --extra gpu` 仍会同步回 cu128。cu126 也不能直接代替支持 Blackwell 所需的构建，见下表。

## 三、驱动要求：兼容下限与部署基线

当前锁文件的 CUDA Runtime 12.8.90 对应 CUDA 12.8 Update 1。新部署采用下列配套驱动版本或更高版本，并确认所选驱动也支持具体显卡型号：

| 平台 | 本项目建议的驱动起点 | CUDA 12.x 小版本兼容的官方驱动下限 |
|---|---:|---:|
| Linux x86_64 | **570.124.06** | **525.60.13** |
| Windows x64 | **572.61** | **528.33** |

左列版本是 CUDA 12.8 Update 1 的配套部署基线；右列是带限制的小版本兼容下限，**不表示所有新显卡、PTX/JIT 或算子都能在该旧驱动上运行**。显卡晚于该驱动发布时，还须满足显卡自身的驱动要求。[CUDA 12.8 Update 1 官方驱动表](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-driver)、[小版本兼容限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

`nvidia-smi` 顶部的 `CUDA Version` 表示驱动支持的最高 CUDA 版本，并不表示当前 Python 使用的运行库版本；`nvcc --version` 表示系统 Toolkit 编译器版本；**当前 PyTorch 的构建版本看 `torch.version.cuda`**。例如 `nvidia-smi` 显示 13.x、PyTorch 显示 12.8 可以正常共存；系统没有 nvcc 也不影响本项目使用 wheel。[NVIDIA nvidia-smi 官方说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

## 四、支持哪些 NVIDIA GPU

以下针对默认 cu128、兼容驱动和显存足够的情况。Compute Capability（CC）是硬件能力号，**不等于 CUDA 软件版本**。型号与能力号依据 [NVIDIA 官方 GPU 表](https://developer.nvidia.com/cuda/gpus)。

| 架构 / CC | 常见型号 | 当前方案与训练精度 |
|---|---|---|
| Turing / 7.5 | T4；RTX 2060、2070、2080、2080 Ti；Quadro RTX 6000/8000 | cu128 范围内；用 `--precision fp16` 或 `none`，不按原生 BF16 卡使用 |
| Ampere / 8.0 | A100、A30 | cu128 范围内；可用 `bf16`、`fp16` 或 `none` |
| Ampere / 8.6 | RTX 3050～3090/3090 Ti；RTX A2000/A4000/A5000/A6000；A10、A40 | cu128 范围内；可用 `bf16`、`fp16` 或 `none` |
| Ada / 8.9 | RTX 4060/4060 Ti、4070/4070 Ti、4080、4090；RTX 6000 Ada；L4、L40、L40S | cu128 范围内；可用 `bf16`、`fp16` 或 `none` |
| Hopper / 9.0 | H100、H200 | cu128 范围内；可用 `bf16`、`fp16` 或 `none` |
| Blackwell / 10.0 | B200、GB200 | cu128 构建覆盖该架构；须使用支持具体型号的驱动并运行自检 |
| Blackwell / 12.0 | RTX 5060/5060 Ti、5070/5070 Ti、5080、5090；RTX PRO 6000 Blackwell | cu128 构建覆盖该架构；须使用支持具体型号的驱动并运行自检 |

这不是型号白名单：同架构的其他型号应结合官方 CC、已安装 wheel 的架构范围和实际自检判断，移动版、显存规格及服务器限制也要分别核对。PyTorch 2.9.1 的 Linux cu128 构建脚本包含 `7.0 / 7.5 / 8.0 / 8.6 / 9.0 / 10.0 / 12.0`；Ada 8.9 可通过同大版本的二进制兼容获得支持，`get_arch_list()` 不必与 GPU 能力号逐字相同。Windows、其他 CPU 架构或不同来源的 wheel 应以其实际构建和自检为准。[固定 v2.9.1 的官方构建脚本](https://github.com/pytorch/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh)、[架构兼容检测实现](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/cuda/__init__.py)

### 旧卡和其他平台

| 硬件 | 当前项目结论 |
|---|---|
| V100 / Volta 7.0 | 上述 2.9.1 Linux cu128 构建含 sm_70；需核对具体 wheel 并自检，不用 BF16。不能用其他 PyTorch 版本的支持列表代替此处的固定版本。 |
| P100、P40、GTX 10 系列等 Pascal；GTX 9 系列等 Maxwell | 默认 cu128 构建已不包含这些架构，不列入当前方案。改用 cu126 或自行编译属于另一个环境方案，需要重新配置和验证。 |
| 更旧的 Kepler 等 GPU | 不列入当前 GPU 方案。 |
| AMD / Intel GPU、Apple GPU | 当前 `--extra gpu` 是 NVIDIA CUDA 方案；没有为 ROCm、XPU、MPS 配置和验证完整训练流程。可选择 CPU 方案。 |
| Jetson、DGX Spark、其他 ARM 设备 | 不从桌面/服务器型号表推定开箱可用；需单独核对平台 wheel、JetPack/驱动及依赖。 |

旧架构范围依据上述 [PyTorch v2.9.1 构建脚本](https://github.com/pytorch/pytorch/blob/v2.9.1/.ci/manywheel/build_cuda.sh)，不同版本的 wheel 会改变该范围。

## 五、FP32、FP16、BF16 怎么选

| 参数 | 当前入口行为与要求 |
|---|---|
| `--precision none` | 普通 FP32 参数及计算流程；默认训练入口使用此值，显存不足时减小微批量 |
| `--precision fp16` | CUDA 自动混合精度并使用 GradScaler；Turing 等没有原生 BF16 的卡可用 |
| `--precision bf16` | CUDA 自动混合精度；优先在具有原生 BF16 支持的 Ampere/Ada/Hopper/Blackwell 上使用，并核对实机支持 |
| `--precision auto` | **仅旧 `train.py` / `resume_train.py` 支持**；CUDA 上按 BF16 支持情况选择，否则使用 FP16。`python -m training` 和独立评估入口不接受 auto |

PyTorch 的 `torch.cuda.is_bf16_supported()` 默认允许考虑模拟支持；需要判断原生支持时可查询 `torch.cuda.is_bf16_supported(including_emulation=False)`。本项目已有自检会打印 BF16 检测结果，仍应按实际训练精度试跑。[PyTorch 2.9.1 CUDA 检测实现](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/cuda/__init__.py)

项目只按单个 `--device cuda` 或 `cuda:0` 执行当前训练流程，没有实现 DDP 多卡训练。服务器有多张卡不会自动把显存合并。

## 六、安装与实机自检

在项目根目录执行，Linux/AutoDL 和 Windows GPU 环境使用同一组 uv 命令：

```bash
uv sync --locked --python 3.12 --extra gpu
nvidia-smi
uv run --locked --extra gpu python tools/check_training_env.py
```

自检会打印解释器、PyTorch、wheel CUDA、可见设备型号、能力号、显存和编译架构，并实际运行 GPU 矩阵乘法与 FP16 SDPA 的前向/反向。主要通过标志为：

```text
PyTorch: 2.9.1+cu128 ... wheel CUDA: 12.8
CUDA available: True
CUDA matmul + SDPA forward/backward: OK; BF16: ...
```

**只有 `CUDA available: True` 还不等于训练可用**，应确认后面的算子前反向也通过。当前自检只对默认 CUDA 设备运行算子测试；使用其他卡或不同精度时须在相应设备上另行核验。它不测完整网络的峰值显存或吞吐。

无 GPU 的本地环境使用：

```bash
uv sync --locked --python 3.12 --extra cpu
uv run --locked --extra cpu python tools/check_training_env.py
```

`--extra cpu` 与 `--extra gpu` 二选一；`neural` 是 CPU 别名。GPU 终端命令持续使用 `uv run --locked --extra gpu ...`，避免运行时同步成 CPU 环境。安装、数据上传和正式训练分别见 [GitHub 与 AutoDL 同步](docs/GitHub与AutoDL同步.md)、[指令.md](指令.md)。

## 七、显存和常见错误

显卡架构兼容不代表任意训练参数都能放入显存。本项目没有经过逐型号测试的最低显存保证，也不将“参数只有数百万”当成显存需求。注意力保存的中间量、反向计算、T、微批量、KV 窗口和精度都会影响峰值。

逻辑批量 `--batch-size 128` 可由多个微批量累积；未知显存余量时可先用 `--micro-batch 1` 试跑，再逐步增加，且须整除逻辑批量。HaiKu 当前默认 KV 窗口为 30；Opus 五个窗口为 10/20/30/20/30；共用训练入口默认每 16 步截断梯度。每个新微批量的 KV 缓存独立重建。

| 现象 | 检查方向 |
|---|---|
| `torch.version.cuda` 为 `None` | 当前是 CPU wheel；检查解释器和是否使用 `--extra gpu` |
| `nvidia-smi` 正常，但 CUDA available 为 False | 驱动、容器 GPU 挂载、可见设备与 Python 环境；安装 Toolkit 本身不能替代驱动 |
| `no kernel image is available` / 架构不兼容警告 | wheel 没覆盖显卡架构，核对 GPU CC 和 wheel；新卡使用对应 cu128 构建，旧卡核对旧架构方案 |
| `driver version is insufficient` / `cudaErrorCallRequiresNewerDriver` | 驱动与所需功能不兼容，按第三节和显卡型号核对 |
| `CUDA out of memory` | 先减小 micro-batch，再按训练目标调整 T/精度；验证也可独立减小其微批量 |
| BF16 / 某个 SDPA 后端报错 | 核对原生精度能力及实际后端；Turing 先使用 fp16 或 none |

如更换 PyTorch、CUDA 索引或锁文件版本，应重新核对本文并在服务器运行自检。当前文档不改变项目依赖版本。
