<div align="center">

# AgInfer

**面向端侧 VLA/VLM 模型的目标相关 AOT 部署方案**

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus&logoColor=white)](CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-sm89%20%7C%20sm110%20%7C%20sm120-76B900.svg?logo=nvidia&logoColor=white)](#支持的目标平台)

[English](README.md) · 简体中文

</div>

AgInfer 将模型 checkpoint 和目标相关的 CUDA 产物编译为单个、可 mmap
加载的 `.aim` 文件。轻量 C++ Runtime 会在部署设备上完成完整性与兼容性校验，
并选择与当前硬件精确匹配的变体；目标设备不依赖 PyTorch、Python、ONNX
Runtime 或 TensorRT。

> [!IMPORTANT]
> AgInfer 只接受精确匹配的目标。Runtime 不会回退到相近 CUDA 架构，也不会
> 执行 PTX JIT、运行时自动调优或运行时权重转换。

## 核心特性

- **单文件部署：** 图、元数据、packed 权重、CUBIN Kernel、tactic 和执行计划
  全部保存在一个 AIM 容器中。
- **架构专用 AOT：** x86_64 包可同时包含 `sm89` 和 `sm120`，Jetson AGX
  Thor 使用独立的 `sm110` 包。
- **安全读取 checkpoint：** 只读取 safetensors 和标准 Hugging Face 元数据，
  拒绝 pickle checkpoint 和远程代码。
- **严格保持 dtype：** 原生支持 FP32、FP16、BF16，以及带显式 scale 的
  ModelOpt Unified Hugging Face E4M3 FP8。
- **默认完整性校验：** 校验整文件和各 section 的 SHA-256、范围、对齐及 ABI。
- **原生集成：** 提供轻量 C++20 Loader 与 Session API，通过显式 `Status`
  返回错误，不依赖异常。

```text
Hugging Face checkpoint ─┐
Shape profile ───────────┼──▶ modelc ──▶ model.aim ──▶ C++ Runtime ──▶ NVIDIA GPU
AOT CUDA 产物 ───────────┘
```

## 目录

- [支持的目标平台](#支持的目标平台)
- [支持的模型](#支持的模型)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [Artifact 目录](#artifact-目录)
- [C++ 集成](#c-集成)
- [Checkpoint 与精度规则](#checkpoint-与精度规则)
- [常见问题](#常见问题)
- [AIM 格式](#aim-格式)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 支持的目标平台

| Host platform | CUDA 架构 | 目标设备 |
|---|---:|---|
| `linux-x86_64-gnu` | `sm89` | GeForce RTX 40 系列 |
| `linux-x86_64-gnu` | `sm120` | GeForce RTX 50 系列 |
| `linux-aarch64-sbsa` | `sm110` | Jetson AGX Thor |

单个 AIM 文件只对应一个 host platform。x86_64 AIM 可以同时包含 `sm89` 和
`sm120`；Thor 必须生成独立的 `linux-aarch64-sbsa + sm110` AIM 文件。

## 支持的模型

| 模型系列 | Adapter | 说明 |
|---|---|---|
| NVIDIA GR00T N1.5 | `groot_n1_5` | Eagle VLM + flow-matching DiT action head |
| LingBot-VLA 1.0 | `lingbot_vla_1` | 无深度版本，内嵌 Qwen2.5-VL backbone |

LingBot Adapter 需要指定 `--backbone <qwen2.5-vl-snapshot>`，从而保证部署时
AIM 文件不依赖外部 backbone。

## 环境要求

### 编译器

- Python 3.10 或更高版本
- 本地 Hugging Face snapshot，或可访问带 immutable revision 的 Hub 仓库
- Shape profile
- 每个目标 CUDA 架构对应的已验证 CUBIN、tactic database 和执行计划

### Runtime

- x86_64 或 aarch64 SBSA Linux
- 支持 C++20 的编译器
- 受支持的 NVIDIA GPU 和兼容的 CUDA Driver
- 与 AIM manifest 匹配的 CUDA Runtime、cuBLASLt 和 cuDNN

## 安装

克隆仓库并安装编译器：

```bash
git clone https://github.com/wenhuiNi/AgInfer.git
cd AgInfer
python3 -m pip install .
```

如需使用 Hugging Face Hub 或 YAML profile，可安装可选依赖：

```bash
python3 -m pip install '.[hub,yaml]'
```

构建 C++ Runtime：

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
cmake --build build -j
```

构建产物为 `libaginfer_runtime.a`。

## 快速开始

### 1. 准备 Shape profile

可以从 [examples/shape-profile.json](examples/shape-profile.json) 开始，分别为
序列长度、图像尺寸与数量、state 长度和 action horizon 设置有界的 `min`、
`opt` 和 `max`。Batch 固定为 1。

### 2. 编译 x86_64 AIM 文件

```bash
modelc compile \
  --source /models/groot-n1.5 \
  --adapter groot_n1_5 \
  --platform linux-x86_64-gnu \
  --cuda-arch sm89 \
  --cuda-arch sm120 \
  --profile examples/shape-profile.json \
  --artifact-dir /models/groot-artifacts \
  --offline \
  --output groot-n1.5.x86_64.sm89-sm120.aim
```

Jetson AGX Thor 需要使用独立目标：

```bash
modelc compile \
  --source /models/groot-n1.5 \
  --adapter groot_n1_5 \
  --platform linux-aarch64-sbsa \
  --cuda-arch sm110 \
  --profile examples/shape-profile.json \
  --artifact-dir /models/groot-thor-artifacts \
  --offline \
  --output groot-n1.5.thor-sm110.aim
```

远程 Hugging Face 源需要安装 `hub` 可选依赖，并指定不可变的十六进制 commit：

```bash
modelc compile \
  --source nvidia/GR00T-N1.5-3B \
  --revision <commit-hash> \
  --adapter groot_n1_5 \
  --platform linux-x86_64-gnu \
  --cuda-arch sm89 \
  --profile examples/shape-profile.json \
  --artifact-dir /models/groot-sm89-artifacts \
  --output groot-n1.5.sm89.aim
```

### 3. 检查 AIM 文件

```bash
modelc inspect groot-n1.5.x86_64.sm89-sm120.aim
```

该命令会先验证 AIM 文件，再输出 platform、CUDA variants、manifest、计算图、
tensor 数量、文件大小和 SHA-256。

## Artifact 目录

`--artifact-dir` 指向 toolchain 元数据，以及每个目标 CUDA 架构的独立目录：

```text
artifacts/
├── toolchain.json
├── sm89/
│   ├── kernels.cubin
│   ├── plan.json
│   └── tactics.json
└── sm120/
    ├── kernels.cubin
    ├── plan.json
    └── tactics.json
```

- `toolchain.json` 声明支持的 CUDA Driver/Runtime 范围，以及要求的 cuBLASLt
  和 cuDNN ABI。
- `kernels.cubin` 必须是与当前目录架构精确匹配的 ELF CUBIN。
- `plan.json` 包含静态 arena、workspace、shape dispatch 和 CUDA Graph template。
- `tactics.json` 包含目标架构上已验证的算法。

任一 artifact 缺失、格式错误或目标架构不匹配时，编译都会返回明确错误。

## C++ 集成

```cpp
#include <aginfer/runtime.h>

aginfer::RuntimeOptions runtime_options;
auto runtime = aginfer::Runtime::Create(runtime_options);
if (!runtime.ok()) {
  // 输出 runtime.status().code() 和 runtime.status().message()。
  return 1;
}

auto model = aginfer::Model::Load("model.aim");
if (!model.ok()) {
  // 处理 model.status()。
  return 1;
}

aginfer::SessionOptions session_options;
session_options.profile = "default";
auto session = aginfer::Session::Create(
    runtime.value(), model.value(), session_options);
if (!session.ok()) {
  // 处理 session.status()。
  return 1;
}

auto target = session.value().GetTargetInfo();
auto workspace = session.value().GetRequiredWorkspace("default");
```

加载 AIM 和创建 Session 时，AgInfer 会依次校验：

1. Host platform 与字节序
2. AIM schema 与 Runtime ABI
3. 整文件与各 section 校验和
4. GPU 的精确 compute capability
5. CUDA Driver 与 CUDA Runtime 兼容性
6. cuBLASLt 与 cuDNN ABI

任一条件不匹配都会返回对应的状态码，不进行 fallback。

## Checkpoint 与精度规则

AgInfer 接受 safetensors、JSON/YAML 配置和标准 tokenizer/processor 元数据。
基于 pickle 的 `.bin`、`.pt`、`.pth`、`.ckpt` 等格式会被拒绝，也不会执行模型
仓库中的远程代码。

FP32、FP16 和 BF16 tensor 会保留 checkpoint dtype。FP8 checkpoint 必须使用
带显式 scale 的 ModelOpt Unified Hugging Face E4M3 格式。编译器不执行校准、
scale 推导或隐式 dtype 转换。

## 常见问题

| 错误 | 排查方法 |
|---|---|
| `INCOMPATIBLE_PLATFORM` | 为 Runtime 所在的 host platform 重新生成 AIM。 |
| `INCOMPATIBLE_ARCHITECTURE` | 编译时加入当前 GPU 精确对应的 `smXX` 变体。 |
| `INCOMPATIBLE_ABI` | 检查 AIM 中声明的 CUDA Driver/Runtime、cuBLASLt、cuDNN 和 Runtime ABI。 |
| `CORRUPT_PACKAGE` | 重新生成或复制 AIM；文件范围、格式或 checksum 校验失败。 |
| Checkpoint 不安全 | 将源权重转换为 safetensors，并移除 pickle 文件。 |
| 缺少 tactic 或 plan | 为每个目标 CUDA 架构提供已验证的 artifact。 |

运行 `modelc --help` 或 `modelc compile --help` 可以查看完整 CLI 参数。

## AIM 格式

[AIM schema 1.0](docs/aim-v1.md) 记录了二进制布局、对齐规则、variant
directory、校验和及兼容性要求。

## 参与贡献

欢迎提交 Issue 和 Pull Request。建议保持改动范围清晰，为行为变化补充测试，
并维持精确目标匹配和明确报错的部署原则。

## 许可证

AgInfer 使用 [MIT License](LICENSE)。
