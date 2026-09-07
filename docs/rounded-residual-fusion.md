# Rounded BF16 residual fusion

`aginfer compile --fuse-residual` replaces an exclusive same-shape BF16
`mul -> add` region with one AOT launch. The product is rounded to BF16 before
the addition, and the sum is rounded to BF16 again. Explicit F32 round-to-nearest
multiply/add intrinsics prevent accidental FMA contraction. This is not a
precision policy change or quantization.

The structural matcher requires contiguous CUDA BF16, rank 1..4, bounded positive
static dimensions (at most 2^26 elements), one private multiply output and exactly
one addition consumer. Shared/exported products, broadcasting and F32 are not
fused. It respects regions already claimed by normalization/GELU fusion and
claims a shared addition only once. Existing topological scheduling and liveness
remove the product buffer. Read inputs may alias; the distinct output cannot
overlap any input. Payload `AIMAD1` binds SM120, numel, module bytes and digest.

The flag is compile-only: ProgramIR, weights, GEMM problems and algorithm
selection do not change. The selected module must contain the new symbol;
missing symbols or bad bindings fail Prepare, never silently fall back. Existing
AIMs and unfused commands remain supported. CUDA Graph remains the public default.

CPU tests check discovery/refusal, prior-fusion ownership, liveness, payload
corruption and both addition operand orders. Optional GPU tests compare the
original two kernels with the fused command at 37, 51200 and 131071 elements,
eager and two poisoned Graph replays, and reject invalid bindings. The largest
local case covers all finite BF16 input bit patterns. A cancellation case
distinguishes the required BF16 product rounding from a direct F32 multiply-add.
Synthetic kernel checks are not substitutes for real model E2E accuracy.
