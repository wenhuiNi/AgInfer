# Experimental source-to-AIM compilation

`aginfer compile` imports a local checkpoint, performs structural provider
lowering, plans command-order storage, packs referenced weights, and writes one
candidate AIM. It does not accept an executable plan or copy commands from an
existing AIM. The only source frontend currently exposed here is the fixed
PI0.5 tensor-input profile on SM120. This is not a general model-support or
production-release claim.

## Build and compile

Build the native library and AOT CUBIN with CMake. Python is a **build-time**
dependency for the compiler and kernel provenance record, not a deployment
dependency:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAGINFER_CUDA_ARCH=120
cmake --build build -j
PYTHONPATH=src python3 -m aginfer.cli select-algorithms CHECKPOINT_DIRECTORY \
  --cubin build/kernels/aginfer-aot-sm120.cubin \
  --selector build/aginfer_cublaslt_select \
  --output fixed-algorithms.json --report selection.json
PYTHONPATH=src python3 -m aginfer.cli compile CHECKPOINT_DIRECTORY \
  --frontend pi05 \
  --cubin build/kernels/aginfer-aot-sm120.cubin \
  --kernel-build-record build/kernels/aginfer-aot-sm120.cubin.build.json \
  --algorithms fixed-algorithms.json --selection-report selection.json --output model.aim
PYTHONPATH=src python3 -m aginfer.cli verify model.aim --require-build-record
PYTHONPATH=src python3 -m aginfer.cli inspect model.aim
```

Compilation is offline. The source must already exist locally. The output
directory must exist and the output file must not; compilation refuses to
overwrite artifacts. `--scratch DIRECTORY` selects scratch storage for weight
packing. Allow space for both the packed weight scratch file and final AIM.
Scratch files are removed after compilation.

The CMake kernel record contains exact CUBIN identity, kernel source hashes,
NVCC executable hash/version, target, flags and the no-PTX inspection result.
It records provenance, not kernel correctness or benchmark results. It is not
a signed attestation or a claim of a hermetic system-header/toolchain build.

## Explicit fixed algorithms

`select-algorithms` derives GEMM descriptors from the source inventory and uses
the offline native `aginfer_cublaslt_select` executable. It does not read an old
AIM, an old algorithm file, model activations, or calibration samples. The helper
queries up to 32 cuBLASLt heuristic candidates for each descriptor, reconstructs
all nine public algorithm configuration fields, and selects the first whose
round-trip and AlgoCheck succeed within the declared workspace limit. It does
**not** benchmark candidates by default or claim the fastest implementation.

`select-algorithms --benchmark-small-gemm` instead enables a bounded offline
prefilter: at most four reconstructable candidates for contiguous, 256-byte
aligned BF16/FP32-accumulate biased GEMMs with batch 1, M=2..128, N/K>=256 and
total device allocation <=128 MiB. Other descriptors retain heuristic-first.
Synthetic nonzero BF16 inputs first require finite, bit-identical output to
the first candidate and a poisoned-output Graph replay check. Three alternating
forward/reverse rounds time an eight-GEMM Graph with CUDA events. A replacement
must win all rounds and lower the median by more than 2%; otherwise the first
candidate remains. Workspace limits and compute types are unchanged.

The v2 probe records candidate ranks, rejected numerical matches, timings and
the selection rule; the old v1 probe remains supported. This hot-buffer seam
measurement is not model accuracy, cold/streaming-weight performance, or E2E
certification. The selection report still says model numerical/capture
validation is not run. Validate changed tactics with real inputs and the same
deployment Graph policy before accepting an AIM. Tuning is never performed in
Load, Prepare or Enqueue. See [NVIDIA's cuBLASLt documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html).

The delivered selection policy uses a 4 MiB workspace limit for linear/patch
GEMMs and zero additional GEMM workspace for materialized attention. F32 linear
uses the fixed TF32 compute policy; patch and F32 vision attention use FP32.
BF16 compute and intermediate-rounding semantics are unchanged. Strided-batch
alignment preferences account for each batch offset, including shared KV inputs
and interleaved BSHD output. `--workspace-limit BYTES` explicitly changes the
linear/patch budget (0..64 MiB); it is recorded in every request. A different
budget may select a different reduction algorithm and change numerical outputs,
so it requires new model validation, not just a successful AlgoCheck.
Selection requires an SM120 device and cuBLASLt
120803. `AGINFER_BUILD_COMPILER_TOOLS=OFF` omits the helper from CUDA builds; it
is never linked into `aginfer_runtime` and is not needed at deployment.

The two output paths must be distinct, new files in existing directories.
`selection.json` binds the helper executable, compiler Python sources, source
IR/inventory, CUBIN, requests, AlgoCheck results, and the canonical algorithm
file. With `compile --selection-report`, these bindings are checked and embedded
in the AIM build record; `verify` checks them again. This records selection,
**not** a model numerical/capture, E2E timing, or signed/hermetic build attestation.
Actual native model execution must validate every newly selected artifact.
Legacy explicit algorithm files can still be compiled without a selection report;
their provenance is not retroactively invented.

`fixed-algorithms.json` remains an explicit, reproducible compilation input.
Its schema is:

```json
{
  "schema": "aginfer.fixed-algorithms.v1",
  "linear": ["<canonical lowercase hex CublasLtLinearPayload>"],
  "patch": "<canonical lowercase hex CublasLtLinearPayload>",
  "attention": [
    "<canonical lowercase hex RoundedAttentionPayload variant 1>",
    "<canonical lowercase hex RoundedAttentionPayload variant 2>",
    "<canonical lowercase hex RoundedAttentionPayload variant 4>"
  ]
}
```

There must be exactly one `linear` payload for every unique linear problem
in the imported program, with no missing or extra problems. `patch` describes
the patch-projection GEMM; its boundary is checked by `PatchProjectionPayload`.
Attention variants cover BF16 denoise, BF16 prefix and FP32 vision respectively.
They bind the exact CUBIN hash/size; changing the module requires new selections
and new validation. The current selection envelope requires cuBLASLt 120803.
Algorithm IDs alone are insufficient: full configuration, compute mode,
alignment and workspace are encoded by the strict public payload classes.
No algorithm is selected, tuned or changed by the runtime.

See [cuBLASLt payload](cublaslt-linear.md) and
[materialized attention](rounded-bf16-attention.md) for the executable forms.
The three attention operations remain two library GEMMs plus one softmax
launch, not a single fused kernel.

## Descriptions are not validation receipts

`BuildBinding` / `LinearBuildBinding` explicitly describe **unvalidated**
candidate implementations. They contain no measured outputs or timings and
produce capabilities with capture support false. Existing measured receipt
classes retain their stricter validation rules. Structural matchers are shared;
the compiler never fabricates a successful receipt to call a matcher.

Each used command capability binds its exact payload digest. The manifest
additionally binds checkpoint assets, compiler Python sources, kernel build
record, fixed algorithms, ProgramIR, inventory, schedule, command memory plan,
packed weights and executable sections. Absolute build-machine paths and
timestamps are excluded from these identities. Changing code or inputs changes
the build identity; identical fixed inputs produce reproducible artifacts.

Compilation emits `unvalidated_candidate` / `validation.status=not_run` even
when a similar previous model passed. Actual model validation is a separate
same-input/same-noise comparison, with execution counts and negative controls,
bound to the **new artifact's raw-file SHA-256**. Do not transfer an old receipt
to a different module or claim numerical success from compilation alone.

## What verify proves

`verify` checks container and section integrity, fully parses execution-plan v2,
decodes supported provider payloads, checks ABI/tag/workspace/module/target,
and cross-checks compiler identity and every used capability when present.
`--require-build-record` refuses older artifacts that lack that record.
Verification does not launch CUDA, rerun cuBLASLt AlgoCheck, run inference,
or certify numerical accuracy, performance, runtime availability or capture.
Older plan-v1 files can still be inspected, but this stronger verifier rejects
them rather than pretending its header-only parser checks a full executable.

Reports distinguish the raw-file SHA-256 from the AIM container checksum,
which zeros its own header checksum field before hashing. Never use the latter
as an unlabelled `sha256sum` value.
