# Exact patch-projection command

AgInfer provides one fail-closed SM120 command for the fixed non-overlapping
vision patch projection

```text
image F32 [1,3,224,224]
weight F32 [1152,3,14,14] + bias F32 [1152]
    -> projected patches F32 [1,16,16,1152]
```

The source ProgramIR expresses this as stride-14, zero-padding `conv2d`
followed by NCHW-to-NHWC transpose. Since the 14x14 patches do not overlap,
the executable form first uses
`aginfer_patchify_f32_nchw_224_p14` to produce the exact row-major
`[256,588]` C/H/W patch matrix, then runs a fixed cuBLASLt
`M=256,N=1152,K=588` F32 matrix multiplication with bias epilogue. cuBLASLt
writes `[256,1152]` row-major physical order directly, which is the required
NHWC output and removes the separate output transpose.

This is one prepared provider command containing two caller-stream operations:
one AOT layout kernel and one exact cuBLASLt algorithm. `Execute` does not
allocate, synchronize, load a module, run a heuristic, select a tactic or use a
fallback. CUTLASS is already present transitively in the pinned FlashInfer
source tree, but is not needed for this F32 cell: the installed cuBLASLt
algorithm passed reconstruction, correctness and capture gates at the real
shape, while the only missing work is the model-independent patch layout
transform.

The structural lowerer requires the exact input/weight/bias/output contracts,
stride, padding, exclusive conv output, same invocation and exact
`(0,2,3,1)` transpose. It emits three commands for the three call-expanded
images, all sharing the same constant weight and bias. Recognition does not use
a model ID, function name, camera index, debug value name or tensor contents.
A changed shape, dtype, layout, stride, padding, consumer set, permutation or
target fails closed.

The 384-byte payload contains a 192-byte patch header followed by the existing
192-byte cuBLASLt payload. Together they bind launch geometry, tensor bounds,
alignment, the 602,112-byte patch matrix, the exact cuBLASLt algorithm and
2,304-byte vendor workspace, target architecture, library version, and the
caller-owned CUBIN size/SHA-256. `Prepare` reconstructs and checks the
cuBLASLt algorithm once, resolves the AOT symbol, partitions the 604,416-byte
caller workspace, and rejects null, short, misaligned or overlapping ranges.

Public tests cover Python/C++ payload agreement, corruption, strict receipt
gates, deterministic two-op structural fusion, public intermediate refusal and
target mismatch. Target-device acceptance uses the frozen E0 image and actual
checkpoint weights, compares all 294,912 outputs to an independent
double-accumulation convolution reference, requires bit-exact repeat and CUDA
Graph replay, a C/H/W-vs-H/W/C patch-order negative control, zero racecheck
hazards, bounds/alias/architecture/module negatives, a PTX-free and
independently reproducible CUBIN, and paired timing against an independent
direct-convolution-plus-transpose CUDA semantic baseline. That baseline is not
the final unmodified-host E2E comparator; the strong host comparison remains an
E2E acceptance gate.
