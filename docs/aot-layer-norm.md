# AOT CUDA LayerNorm provider v1

The built-in AOT CUDA provider currently exposes one exact affine LayerNorm
problem:

```text
input/output  [1, 256, 1152]  row-major F32 CUDA
weight/bias   [1152]          row-major F32 CUDA
epsilon       1e-6
target        SM120
```

`kernels/layer_norm.cu` exports `aginfer_layer_norm_f32_1152`. One 256-thread
block handles each of 256 rows. It first reduces the row mean, then performs a
second centered-variance reduction before writing the affine result. The
kernel is part of the same target-specific, PTX-free CUBIN as the other
built-in AOT kernels.

The 192-byte numeric payload binds the exact rows, width, dtype, layout,
epsilon, launch geometry, alignments, tensor byte sizes, CUBIN byte size, and
CUBIN SHA-256. It contains no pointers or symbol strings.

`AotLayerNormCommand::Prepare` validates the payload, active architecture,
caller-owned module identity, all four tensor bindings, and output
non-aliasing, then resolves the fixed symbol. `Execute` performs one driver
launch on the caller stream. It does not allocate, synchronize, load modules,
select a tactic, or fall back.

Other row counts, widths, dtypes, epsilon values, layouts, in-place forms, and
architectures remain unresolved at compile time.
