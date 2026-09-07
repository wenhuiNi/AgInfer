# cuBLASLt linear provider payload v1

This provider boundary lowers the generic ProgramIR operation

```text
Y[..., N] = X[..., K] * transpose(W[N, K]) + bias[N]
```

to an exact cuBLASLt matrix-multiplication problem. It is structural and does
not use a model ID, function name, or checkpoint slot name. All tensors must be
CUDA, row-major, positive static shapes, and one exact `f32` or `bf16` dtype.
The leading dimensions of `X` are checked and flattened into `M`. cuBLASLt's
bias epilogue uses the physical D-row count, so the row-major buffers are bound
as column-major transposed views: `D[N,M] = W[K,N]^T * X[K,M]`. This produces
the same row-major `Y[M,N]` bytes while making the physical D-row count, and
therefore the bias length, exactly `N`.

FP32 accumulation is the default. An offline payload may explicitly select
`CUBLAS_COMPUTE_32F_FAST_TF32` for F32 tensors when the source precision
contract permits it. Its algorithm and numerical receipt are separate from
the FP32 form; the runtime never infers the mode from a model name or device.
Existing FP32 payload bytes remain unchanged. BF16 with TF32 mode is refused.

The corresponding tagged command has exactly four operands in this order:

1. `X`, read;
2. `W`, read;
3. `bias`, read;
4. `Y`, write.

The 192-byte provider payload contains no pointer, string, or raw
`cublasLtMatmulAlgo_t`. It uses explicitly little-endian fields:

| Offset | Type | Field |
|---:|---|---|
| 0 | `u8[8]` | magic `AILTMM1\0` |
| 8 | `u16` | schema major, currently 1 |
| 10 | `u16` | schema minor: 0 for FP32, 1 for explicit TF32 |
| 12 | `u32` | payload size, exactly 192 |
| 16 | `u32` | exact CUDA architecture |
| 20 | `u32` | exact `cublasLtGetVersion()` result |
| 24 | `u32` | tensor dtype: `f32=1`, `bf16=2` |
| 28 | `u32` | compute type: FP32=1; FAST_TF32=2 (F32 tensors only) |
| 32 | `u32` | scale type, fixed FP32 |
| 36 | `u32` | flags, fixed bias-present bit |
| 40 | `u32` | epilogue, fixed bias |
| 44 | `u32` | A/weight transform, fixed transpose |
| 48 | `u32` | B/X transform, fixed identity |
| 52 | `u32` | A order, fixed column-major view |
| 56 | `u32` | B order, fixed column-major view |
| 60 | `u32` | C order, fixed column-major view |
| 64 | `u32` | D order, fixed column-major view |
| 68 | `u64` | M |
| 76 | `u64` | N |
| 84 | `u64` | K |
| 92 | `u64` | lda, exactly K |
| 100 | `u64` | ldb, exactly K |
| 108 | `u64` | ldc, exactly N |
| 116 | `u64` | ldd, exactly N |
| 124 | `u64` | exact checked workspace bytes |
| 132 | `i32` | algorithm ID |
| 136 | `u32` | tile ID |
| 140 | `i32` | non-negative split-K configuration; zero is preserved |
| 144 | `u32` | reduction scheme |
| 148 | `u32` | CTA swizzling |
| 152 | `u32` | custom option |
| 156 | `u32` | stages ID |
| 160 | `u16` | inner-shape ID |
| 162 | `u16` | cluster-shape ID |
| 164 | `u32` | X pointer alignment |
| 168 | `u32` | weight pointer alignment |
| 172 | `u32` | bias pointer alignment |
| 176 | `u32` | output pointer alignment |
| 180 | `u32` | workspace pointer alignment |
| 184 | `u8[8]` | reserved, zero |

Alignments are bounded powers of two. Matrix byte-size multiplication is
checked against `u64`. Unknown schema/target/dtype/fixed fields, inconsistent
leading dimensions, zero dimensions, invalid algorithm integers, invalid
alignments, reserved bytes, and any non-exact payload size fail closed in both
the Python and C++ parsers.

The payload intentionally has no independent checksum. It is opaque content of
a tagged command and is therefore covered by the command-stream body SHA-256;
the future AIM section digest covers the command header as well. Changing a
syntactically free algorithm field can still produce another syntactically valid
payload, so only an offline receipt from the exact cuBLASLt version, target GPU,
`cublasLtMatmulAlgoCheck`, a real launch, correctness comparison, and capture
test may create a provider capability or executable command. Codec round trips
alone are not that receipt.

## Validation receipt and partial lowering

`CublasLtValidationReceipt` is the fail-closed input from that offline probe. Its
canonical record contains the exact 192-byte payload as lowercase hex and its
SHA-256, exact CUDA driver/runtime identities, successful AlgoCheck, at least
two normal launches with bit-identical output, bit-identical graph capture
replay, at least 64 CPU samples, and the locked sampled error tolerance
(`5e-4` for F32 and `5e-3` for BF16). Unknown, missing, or extra fields fail;
false evidence flags, non-finite metrics, a relaxed tolerance, error above the
gate, malformed payload, and a payload digest mismatch also fail.

One validated receipt can create one exact `ProviderCapability` for every
static site with the same semantic signature. The capability uses provider ID
1, ABI 1.0, and implementation ID `cublaslt.linear.bias.v1`. Its required
`implementation_digest` is the provider payload SHA-256, so the capability
digest embedded in each command is bound to the selected algorithm and not just
to its workspace size.

`lower_cublaslt_linear_commands(...)` emits only the matching scheduled linear
operations. Each command has numeric X/W/bias/Y operands with
read/read/read/write access, the exact validated payload, shared workspace
offset zero, and the exact capability digest. It reports elided memory-plan ops
and every unhandled execution index separately. It deliberately does not emit a
complete command stream while another scheduled operation remains unhandled.

## Native prepared command

The internal native seam is part of the default CUDA build. A deliberately
reduced contract-only build can disable all CUDA providers with
`AGINFER_ENABLE_CUDA=OFF`. `CublasLtLinearCommand::Prepare` parses the payload,
requires an exact active CUDA architecture and
`cublasLtGetVersion()`, validates tensor and workspace pointer alignments, and
rejects missing or short workspace. It builds the fixed descriptors, restores
every recorded algorithm attribute through `cublasLtMatmulAlgoInit` and config
setters, and requires `cublasLtMatmulAlgoCheck` to report exactly the recorded
workspace size.

The prepared command freezes X, weight, bias, output, and workspace addresses.
It does not own those buffers. `Execute` accepts only the caller stream and
issues one `cublasLtMatmul` with the reconstructed algorithm; there is no
heuristic query, allocation, synchronization, address lookup, or fallback in
that path. Descriptor and handle destruction happens in reverse dependency
order when the prepared command is released. Session-level command dispatch,
buffer-size validation, and ledger accounting remain later runtime work.
