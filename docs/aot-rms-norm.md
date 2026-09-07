# AOT CUDA RMSNorm provider v1

The built-in AOT CUDA provider exposes two exact RMSNorm problems:

```text
variant A
input/output  [1, 50, 1024]    row-major F32 CUDA
weight        [1024]           row-major F32 CUDA

variant B
input/output  [1, 968, 2048]   row-major BF16 CUDA
weight        [2048]           row-major F32 CUDA

both variants
epsilon       1e-6
target        SM120
```

Both variants implement `x * rsqrt(mean(x * x) + epsilon) * weight` with F32
accumulation. `kernels/rms_norm.cu` exports one fixed symbol per variant. One
256-thread block handles each row; the F32 variant launches 50 blocks and the
BF16 variant launches 968 blocks. The kernels are part of the same
target-specific, PTX-free CUBIN as the other built-in AOT kernels.

The BF16 form preserves separate F32 square and addition rounding, using
four accumulators per lane and ascending warp-shuffle reduction. This
follows the contiguous F32 mean reduction order in PyTorch 2.9's
`ATen/native/cuda/Reduce.cuh`; no PyTorch code or library is linked at runtime.
The affine weight remains F32 and only the final output is rounded to BF16.
Changing the reduction tree can change BF16 halfway cases, so a new module
digest requires renewed numerical validation.

The 192-byte numeric payload binds a variant ID, rows, width, activation and
weight dtypes, layout, epsilon, launch geometry, alignments, tensor byte sizes,
CUBIN byte size, and CUBIN SHA-256. It contains no pointers or symbol strings.

`AotRmsNormCommand::Prepare` validates the payload, active architecture,
caller-owned module identity, all three tensor bindings, and output
non-aliasing, then resolves the fixed symbol. `Execute` performs one driver
launch on the caller stream. It does not allocate, synchronize, load modules,
select a tactic, convert dtypes, or fall back.

The BF16 problem deliberately keeps its F32 affine weight. Other row counts,
widths, activation or weight dtypes, epsilon values, layouts, in-place forms,
and architectures remain unresolved at compile time.
