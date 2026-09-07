# AOT CUDA activation provider v1

The built-in AOT CUDA activation provider exposes four exact contiguous
ProgramIR problems:

```text
BF16  [1, 50, 4096]       approximate=tanh  SM120
F32   [1, 256, 4304]      approximate=tanh  SM120
BF16  [1, 968, 16384]     approximate=tanh  SM120
F32   [1, 1024]           silu              SM120
```

`kernels/activation.cu` computes
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`. Arithmetic is F32; the BF16
form converts on input and rounds once on output. The implementation follows
the established tanh-GELU numerical form and 256-thread elementwise launch
used by mature CUDA inference kernels, while retaining an AgInfer-owned ABI.
No external framework or kernel library is required by this command.
The SiLU form computes `x/(1+exp(-x))` in F32.

The fixed symbols are `aginfer_gelu_tanh_f32`,
`aginfer_gelu_tanh_bf16`, and `aginfer_silu_f32`. They are included in the same
target-specific, PTX-free CUBIN as the other built-in AOT kernels.

The command uses the shared 128-byte numeric CUDA-kernel payload. Numeric
kernel IDs 9 through 11 bind the operation, dtype, exact supported element
count, canonical launch, pointer alignments, CUBIN byte size, and CUBIN
SHA-256. Exact tensor
shape and semantic attributes remain part of the capability digest. Other
shapes, dtypes, approximation modes, layouts, in-place forms, and
architectures remain unresolved at compile time.

`AotActivationCommand::Prepare` validates the payload, active architecture,
caller-owned module identity, input/output bindings, non-aliasing, and fixed
symbol. `Execute` performs one driver launch on the caller stream. It does not
allocate, synchronize, load modules, select a tactic, convert weights, or
fall back.

This is deliberately a standalone activation seam. A future GEMM+GELU fused
epilogue is a different executable form and must carry its own provider
contract and end-to-end evidence before replacing it.
