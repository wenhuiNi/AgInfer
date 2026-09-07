<div align="center">

# AgInfer

**Experimental foundations for target-specific VLA/VLM AOT deployment**

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus&logoColor=white)](CMakeLists.txt)

English · [简体中文](README.zh-CN.md)

</div>

AgInfer is building an offline model compiler and a standalone C++/CUDA
runtime. The intended deployment artifact contains the lowered program,
packed weights, target-specific implementations, and validated execution
metadata, so the deployment process does not need Python, PyTorch, ONNX
Runtime, or TensorRT.

## Project status

This repository is an early implementation foundation. It does **not** yet
provide a released model importer, optimizer, quantizer, production artifact
schema, or end-to-end supported model. GR00T and LingBot are therefore not
advertised as supported models.

The code currently provides:

- bounded safetensors header inspection with pickle rejection;
- namespace-aware source manifests, bounded constant access, and fail-closed
  recipe coverage for single-file or sharded safetensors;
- an experimental, mmap-friendly AIM container with bounds checks; full SHA-256
  verification runs offline or in Debug/verification runtime builds (see
  [load-time checksum policy](docs/runtime-contract.md#load-time-checksum-policy));
- experimental v1 kernel-launch and v2 tagged-provider execution plans;
- an offline source-to-AIM candidate compiler with explicit fixed algorithms,
  content-addressed build records and a v2 executable/payload verifier;
- a minimal typed-SSA ProgramIR verifier, deterministic text dump, and
  pure-Python CPU reference executor for the initial generic op set;
- a C ABI with opaque handles, numeric ports, explicit prepare/bind/enqueue,
  plus a C++17 RAII wrapper;
- a CUDA Driver launch path exercised by a real target-architecture fixture;
- a default CUDA build that produces a target-specific, PTX-free AOT cast,
  pointwise, exact tanh-GELU/SiLU, F32 vision-attention, LayerNorm, RMSNorm,
  split-half RoPE, denoise KV-pack, prefix KV-store, denoise metadata, and
  prefix-input assembly, and patchify CUBIN,
  with native prepared commands plus the exact cuBLASLt linear provider;
- an exact SM120 FlashInfer FA2 command for the PI0.5-style BF16 GQA denoise
  attention region, with dense 2-D BOOL mask and fused BSHD output;
- an exact SM120 FlashInfer FA2 command for the PI0.5-style BF16 GQA prefix
  attention region, consuming the shared 1-D pad mask, preserving finite-mask
  fully-masked-row semantics, and fusing four mask operations plus BSHD output;
- an exact SM120 built-in AOT command for the PI0.5-style F32 vision-attention
  region, fusing its input/output transposes and scalar-mask broadcast;
- an exact SM120 built-in AOT affine LayerNorm command for the F32
  `[1,256,1152]` vision boundary;
- exact SM120 built-in AOT RMSNorm commands for the F32 `[1,50,1024]` and
  BF16-activation/F32-weight `[1,968,2048]` boundaries;
- an exact SM120 fused adaptive RMSNorm command for BF16 `[1,50,1024]`
  hidden state plus F32 `[1,3072]` scale/shift/gate modulation, replacing the
  exclusive cast/norm/slice/broadcast/pointwise chain with one launch;
- an exact SM120 denoise KV-pack command that joins prefix/current BF16 K/V and
  absorbs the one-head BSHD-to-BHSD V transpose in one launch;
- an exact SM120 prefix KV state-store command that writes both persistent
  states in one launch, with singleton-axis row-major transposes proven as views;
- an exact SM120 denoise suffix-metadata command that expands the prefix pad
  mask and constructs all 50 position IDs in one launch while preserving mask holes;
- an exact SM120 prefix-input command that gathers/scales the language
  embedding, concatenates three projected images, and constructs the shared
  pad mask and position IDs in one launch;
- an exact SM120 patch-projection command that combines an AOT NCHW patchify
  transform with a locked cuBLASLt F32 projection and writes NHWC directly;
- an exact SM120 F32 time sinusoidal-embedding command that preserves the
  source model's Float64 period/trigonometric semantics in one launch;
- an exact SM120 terminal F32 action-slice command that copies columns
  `[0,7)` from every 32-wide row without pretending the result is an alias;
- deterministic compile-time materialization of static BOOL/I32/F32/BF16
  literal broadcasts into a checked, content-deduplicated constant blob, plus
  zero-copy planning for broadcasts that only insert singleton axes;
- exact SM120 built-in split-half BF16 RoPE commands for sequence lengths
  968/50 and 8/1 heads, fusing each exclusive BSHD-to-BHSD input transpose;
- exact SM120 built-in tanh-GELU commands for the three delivered BF16/F32
  activation shapes and an F32 SiLU command for `[1,1024]`;
- corruption, target mismatch, tensor-contract, and launch-plan tests.

The current AIM schema and runtime ABI are experimental and may change without a
compatibility guarantee. SHA-256 detects accidental corruption; it does not
authenticate an artifact from an untrusted source.

The implemented ownership, versioning, and execution rules are documented in
[the experimental runtime contract](docs/runtime-contract.md).
The independently versioned semantic layer is described in the
[experimental ProgramIR contract](docs/program-ir.md).
The offline checkpoint inventory and constant addressing rules are documented
in the [experimental source contract](docs/source-contract.md).
The first source-only recipe audit is recorded in the
[PI0.5 source inventory](docs/pi05-source-inventory.md); it is not an
end-to-end model-support claim.
The three currently delivered attention regions are described by the
[FlashInfer contract](docs/flashinfer-attention.md) and the
[built-in F32 vision-attention contract](docs/aot-vision-attention.md).
The current normalization boundaries are documented in the
[built-in F32 LayerNorm contract](docs/aot-layer-norm.md) and the
[built-in RMSNorm contract](docs/aot-rms-norm.md). The fused denoise
normalization region is specified by the
[adaptive RMSNorm contract](docs/aot-adaptive-rms-norm.md).
The exact denoise K/V materialization boundary is specified by the
[KV-pack contract](docs/aot-kv-pack.md).
The paired persistent prefix-cache write is specified by the
[prefix KV state-store contract](docs/aot-prefix-kv-store.md).
The denoise mask/position construction region is specified by the
[suffix-metadata contract](docs/aot-suffix-metadata.md).
The prefix embedding/mask/position assembly region is specified by the
[prefix-input contract](docs/aot-prefix-input.md).
The non-overlapping vision patch projection is specified by the
[patch-projection contract](docs/patch-projection.md).
The fixed Float64-semantic timestep expansion is specified by the
[time-embedding contract](docs/aot-time-embedding.md).
The terminal tensor-only output boundary is specified by the
[action-slice contract](docs/aot-action-slice.md).
The offline literal-broadcast encoding and memory-plan integration are specified
by the [compile-time materialization contract](docs/literal-materialization.md).
The fixed split-half rotary regions are documented in the
[built-in RoPE contract](docs/aot-rope.md).
The fixed activation regions are documented in the
[built-in activation contract](docs/aot-activation.md).

## Build and test

Run the Python contract tests without installing the package:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests
```

Build the default CUDA runtime, native providers, and target-specific AOT
kernel bundle, then run the C++ contract tests. The default target is `sm120`:

```bash
git submodule update --init third_party/flashinfer
git -C third_party/flashinfer submodule update --init 3rdparty/cccl
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Only the pinned FlashInfer and CCCL sources are consumed; the deployment
runtime has no Python/JIT dependency. See the
[exact attention provider contract](docs/flashinfer-attention.md).

Select another supported deployment architecture explicitly with
`-DAGINFER_CUDA_ARCH=89` or `110`. A contract-only build without the CUDA
Toolkit/provider link is available as an intentional opt-out:

```bash
cmake -S . -B build-contract -DAGINFER_ENABLE_CUDA=OFF
cmake --build build-contract -j
ctest --test-dir build-contract --output-on-failure
```

The repository tests do not claim model-level correctness or performance.

On a supported GPU, the direct CUDA execution cell can be enabled explicitly;
the architecture must match the active device:

```bash
cmake -S . -B build-gpu -DAGINFER_BUILD_CUDA_TESTS=ON \
  -DAGINFER_CUDA_ARCH=120
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

## Artifact inspection

Install the Python package to expose the artifact inspection command:

```bash
python3 -m pip install .
aginfer inspect model.aim
```

`aginfer inspect` validates the experimental container before displaying its
platform, CUDA variants, manifest, graph metadata, tensor count, size, and
digest. Experimental `aginfer select-algorithms`, `aginfer compile`, and `aginfer verify` are documented in
[candidate compilation](docs/candidate-compilation.md). Compilation requires
an explicit offline algorithm selection, obtainable with the native AlgoCheck
helper, and produces an unvalidated candidate. Selection is not benchmark tuning;
it is not a numerical certification or production model-support promise.

Inspect a local checkpoint without loading tensor payloads or framework code:

```bash
aginfer source-manifest /path/to/checkpoint --offline --output /tmp/source.json
aginfer source-contract /path/to/checkpoint --offline
```

The second command reports typed model IO, ordered processor steps, referenced
state assets, and unresolved external asset requirements such as tokenizers.

## Runtime boundary

The deployment runtime is intended to consume only pre-lowered execution
artifacts. Model import, graph rewriting, calibration, quantization, tactic
selection, weight transformation, and performance qualification belong to
the offline build process. The runtime must not silently fall back, JIT
compile, autotune, or repack weights.

## License

AgInfer is released under the [MIT License](LICENSE).
