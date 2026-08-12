<div align="center">

# AgInfer

**Target-specific AOT deployment for edge VLA/VLM models**

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus&logoColor=white)](CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-sm89%20%7C%20sm110%20%7C%20sm120-76B900.svg?logo=nvidia&logoColor=white)](#supported-targets)

English · [简体中文](README.zh-CN.md)

</div>

AgInfer compiles model checkpoints and target-specific CUDA artifacts into a
single mmap-friendly `.aim` file. Its lightweight C++ runtime validates and
selects the exact package variant for the deployment machine—without PyTorch,
Python, ONNX Runtime, or TensorRT on the target.

> [!IMPORTANT]
> AgInfer uses exact target matching. It never falls back to a nearby CUDA
> architecture, PTX JIT, runtime autotuning, or runtime weight conversion.

## Highlights

- **Single-file deployment:** graph, metadata, packed weights, CUBIN kernels,
  tactics, and execution plans live in one AIM container.
- **Architecture-specific AOT:** package `sm89` and `sm120` together for x86_64,
  or build a dedicated `sm110` package for Jetson AGX Thor.
- **Safe checkpoint ingestion:** safetensors and standard Hugging Face metadata
  only; pickle checkpoints and remote code are rejected.
- **Strict dtype preservation:** native FP32, FP16, BF16, and ModelOpt Unified
  Hugging Face E4M3 FP8 with explicit scales.
- **Integrity by default:** whole-file and per-section SHA-256 verification,
  bounds checks, alignment checks, and ABI validation.
- **Native execution:** a C++20 Runtime loads CUBIN directly through the CUDA
  Driver API and enqueues the static launch recipe on the caller's stream.

```text
Hugging Face checkpoint ─┐
Shape profile ───────────┼──▶ modelc ──▶ model.aim ──▶ C++ Runtime ──▶ NVIDIA GPU
AOT CUDA artifacts ──────┘
```

## Table of contents

- [Supported targets](#supported-targets)
- [Supported models](#supported-models)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Artifact layout](#artifact-layout)
- [C++ integration](#c-integration)
- [Checkpoint and precision policy](#checkpoint-and-precision-policy)
- [Troubleshooting](#troubleshooting)
- [AIM format](#aim-format)
- [Contributing](#contributing)
- [License](#license)

## Supported targets

| Host platform | CUDA architecture | Target devices |
|---|---:|---|
| `linux-x86_64-gnu` | `sm89` | GeForce RTX 40 series |
| `linux-x86_64-gnu` | `sm120` | GeForce RTX 50 series |
| `linux-aarch64-sbsa` | `sm110` | Jetson AGX Thor |

One AIM file targets exactly one host platform. An x86_64 file may contain both
`sm89` and `sm120`; Thor requires a separate `linux-aarch64-sbsa + sm110` file.

## Supported models

| Model family | Adapter | Notes |
|---|---|---|
| NVIDIA GR00T N1.5 | `groot_n1_5` | Eagle VLM with a flow-matching DiT action head |
| LingBot-VLA 1.0 | `lingbot_vla_1` | Depth-free variant with an embedded Qwen2.5-VL backbone |

The LingBot adapter requires `--backbone <qwen2.5-vl-snapshot>` so the deployed
AIM file has no external backbone dependency.

## Requirements

### Compiler

- Python 3.10 or newer
- A local Hugging Face snapshot, or network access to an immutable Hub revision
- A shape profile
- Verified CUBIN kernels, tactic databases, and execution plans for every
  requested CUDA architecture

### Runtime

- Linux on x86_64 or aarch64 SBSA
- A C++20 toolchain
- A supported NVIDIA GPU and compatible CUDA Driver
- CUDA Runtime, cuBLASLt, and cuDNN versions matching the AIM manifest

## Installation

Clone the repository and install the compiler:

```bash
git clone https://github.com/wenhuiNi/AgInfer.git
cd AgInfer
python3 -m pip install .
```

Install optional Hugging Face Hub and YAML support when needed:

```bash
python3 -m pip install '.[hub,yaml]'
```

Build the C++ runtime library:

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
cmake --build build -j
```

The build produces `libaginfer_runtime.a`.

## Quick start

### 1. Prepare a shape profile

Start with [examples/shape-profile.json](examples/shape-profile.json) and set
bounded `min`, `opt`, and `max` values for sequence length, image size/count,
state length, and action horizon. Batch size is fixed to 1.

### 2. Compile an x86_64 AIM file

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

For Jetson AGX Thor, use a dedicated target:

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

Remote Hugging Face sources require the `hub` extra and an immutable hexadecimal
commit hash:

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

### 3. Inspect and verify the package

```bash
modelc inspect groot-n1.5.x86_64.sm89-sm120.aim
```

The command verifies the AIM file before printing its platform, CUDA variants,
manifest, graph, tensor count, file size, and SHA-256 digest.

## Artifact layout

`--artifact-dir` points to toolchain metadata and one directory per requested
CUDA architecture:

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

- `toolchain.json` declares the supported CUDA Driver/Runtime range and the
  required cuBLASLt and cuDNN ABIs.
- `kernels.cubin` must be an ELF CUBIN for the exact directory architecture.
- `plan.json` contains the static arena, workspace, shape dispatch, and CUDA
  Graph template data.
- `tactics.json` contains verified algorithms for the target architecture.

Compilation stops with an explicit error when an artifact is missing,
malformed, or targets a different architecture.

## C++ integration

```cpp
#include <aginfer/runtime.h>

aginfer::RuntimeOptions runtime_options;
auto runtime = aginfer::Runtime::Create(runtime_options);
if (!runtime.ok()) {
  // Report runtime.status().code() and runtime.status().message().
  return 1;
}

auto model = aginfer::Model::Load("model.aim");
if (!model.ok()) {
  // Report model.status().
  return 1;
}

aginfer::SessionOptions session_options;
session_options.profile = "default";
auto session = aginfer::Session::Create(
    runtime.value(), model.value(), session_options);
if (!session.ok()) {
  // Report session.status().
  return 1;
}

auto target = session.value().GetTargetInfo();
auto input_info = session.value().GetInputInfo();
auto output_info = session.value().GetOutputInfo();
auto workspace = session.value().GetRequiredWorkspace("default");

// Allocate and populate device buffers from input_info/output_info.
std::vector<aginfer::TensorView> inputs{/* populated TensorViews */};
std::vector<aginfer::TensorView> outputs{/* populated TensorViews */};
aginfer::RunOptions run_options;
aginfer::CudaStream stream = nullptr;  // CUDA default stream, or a caller-owned stream.
auto status = session.value().Enqueue(inputs, outputs, run_options, stream);
```

During loading and session creation, AgInfer validates:

1. Host platform and byte order
2. AIM schema and Runtime ABI
3. Whole-file and section checksums
4. Exact GPU compute capability
5. CUDA Driver and Runtime compatibility
6. cuBLASLt and cuDNN ABIs

Any mismatch returns a specific status code; no fallback is attempted.
The first `Enqueue` initializes the selected CUBIN module and static device
allocations. Kernel launches are then submitted asynchronously to the supplied
stream; synchronization remains under caller control.

## Checkpoint and precision policy

AgInfer accepts safetensors, JSON/YAML configuration, and standard
tokenizer/processor metadata. Pickle-based `.bin`, `.pt`, `.pth`, `.ckpt`, and
related formats are rejected, and remote model code is never executed.

FP32, FP16, and BF16 tensors preserve their checkpoint dtype. FP8 checkpoints
must use ModelOpt Unified Hugging Face E4M3 with explicit scales. Calibration,
scale inference, and implicit dtype conversion are not performed.

## Troubleshooting

| Error | What to check |
|---|---|
| `INCOMPATIBLE_PLATFORM` | Build the AIM file for the runtime host platform. |
| `INCOMPATIBLE_ARCHITECTURE` | Include the GPU's exact `smXX` variant during compilation. |
| `INCOMPATIBLE_ABI` | Match the CUDA Driver/Runtime, cuBLASLt, cuDNN, and Runtime ABI declared by the AIM file. |
| `CORRUPT_PACKAGE` | Rebuild or re-copy the AIM file; its bounds, format, or checksum validation failed. |
| Unsafe checkpoint error | Convert the source weights to safetensors and remove pickle files. |
| Missing tactic or plan | Provide verified artifacts for every requested CUDA architecture. |

Run `modelc --help` or `modelc compile --help` for the complete CLI reference.

## AIM format

See [AIM schema 1.0](docs/aim-v1.md) for the container layout and
[Kernel ABI 1.0](docs/kernel-abi-v1.md) for binary execution plans, tensor
contracts, and CUDA launch arguments.

## Contributing

Issues and pull requests are welcome. Please keep changes focused, include
tests for behavior changes, and preserve the exact-target and fail-closed
deployment guarantees.

## License

AgInfer is released under the [MIT License](LICENSE).
