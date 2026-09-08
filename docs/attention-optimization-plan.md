# Non-quantized attention / TMA roadmap

This is an audited plan, not a delivered attention speedup. The current
materialized provider remains the numerical baseline.

FlashAttention-3 targets Hopper H100/H800. Its TMA-based loading,
producer/consumer warp specialization and overlap of GEMM with softmax are
useful design references; its FP8 path is outside the current precision policy.
See the [official README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)
and [author's technical explanation](https://tridao.me/blog/2024/flash3/).

TMA support does not imply a usable BF16 attention artifact. The pinned CUTLASS
source enables `CUTE_ARCH_TMA_SM120_ENABLED`, but its SM120 dense TMA collective
builder currently asserts F8F6F4 input types. The
[NVIDIA SM120 documentation](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)
describes these narrow-precision forms separately from SM100. Do not silently
quantize BF16 to obtain a supported builder, or assume SM100/Hopper instructions
and multi-SM schedules work on SM120. The pinned FlashInfer Hopper prefill
entry points also explicitly reject custom masks in the audited paths.

## Ordered acceptance

1. Probe a valid SM120 BF16 TMA copy path at real Q/K/V tile shapes before
   implementing an attention provider. Compare vector/cp.async staging with
   TMA, including descriptor setup cost, shared memory, barriers and occupancy.
   Require byte-exact tiles, tail/alignment/swizzle checks and Graph replay.
2. Target denoise Q `[1,8,50,256]`, KV `[1,1,1018,256]`, GQA8, then prefix
   length968. Explicitly support actual pad-mask holes and fully masked rows.
   Keep F32 vision/head-dim72 outside this change.
3. Preserve or explicitly gate changes to the materialized BF16 QK and
   probability rounding. Online softmax is not automatically bit-equivalent.
   Use real model seam inputs, negative mask controls, then unchanged official
   action gates, repeat/changed-input controls, target-call ledger and zero
   fallback. Compare complete E2E under identical Graph policy.
4. Integrate only a fixed-version native C++/CUDA AOT executable form. AIM
   records tensor-map shapes/strides; Prepare may relocate descriptors to its
   fixed buffers. Enqueue must not allocate, build descriptors, search tactics
   or JIT. A descriptor's device addresses cannot be persisted across processes.

TMA alone is not a reason to replace a global-to-global KV copy with two extra
transfers through shared memory. Prioritize tiles with actual compute reuse.
No FA3/TMA runtime path, performance gain or Hopper validation is claimed here.

## Delivered staging probe

`tools/tma_probe.cu` is a standalone, opt-in native SM120 experiment. Build with
the nvcc command at its top (SASS only, no JIT); it is not linked into deployment
or run by the default test suite. It compares vector, cp.async and TMA loading
into two shared-memory buffers at lengths50/968/1018, width256, tile16/32x64,
eight repeated consumers and no/128-byte swizzle. Four steps reuse each barrier
with alternating parity. The output includes every loaded tile so byte parity
and hardware OOB zero fill are observable. Misaligned maps/strides are refused;
eager and changed/restored poisoned Graph outputs are checked before timing.

On the available RTX5090/CUDA12.8, all 12 cells passed. The SM120 TMA copy
mechanism is therefore executable independently of CUTLASS's narrow-precision
GEMM builders. These staging-only measurements did not show a consistent win
over cp.async; retain both options when adding actual shared-memory computation.
The short timing includes output stores and shared-memory synchronization, not
attention math. Some first-arm samples were outliers. This is neither a DRAM
bandwidth measurement, a race-sanitizer attestation, nor model E2E evidence.
Tensor-map construction is outside timing, as it would be in Prepare; this
probe does not measure descriptor initialization cost or any MMA overlap.

Reference: [NVIDIA asynchronous-copy synchronization](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html)
and [tensor-map API constraints](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html).
