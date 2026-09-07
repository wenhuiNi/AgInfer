# AOT CUDA cast provider v1

The built-in AOT CUDA provider implements three contiguous ProgramIR cast
semantics:

- `bf16 -> f32`;
- `f32 -> bf16`, using round-to-nearest-even;
- `bool -> i32`, where every zero byte becomes 0 and every nonzero byte becomes
  1.

These are product kernels in `kernels/cast.cu`, not test fixtures. A default
CUDA build compiles them directly into the shared target-specific AOT CUBIN.
The CUBIN has no PTX fallback; its cast subset exports these audited symbols:

```text
aginfer_cast_bf16_to_f32
aginfer_cast_f32_to_bf16
aginfer_cast_bool_to_i32
```

The command payload never stores those strings. It carries a numeric kernel ID,
and the native provider maps that ID to the fixed symbol inventory during
`Prepare`.

## Payload

The fixed 128-byte little-endian payload is shared by the cast, pointwise, and
activation providers and contains:

| Offset | Type | Field |
|---:|---|---|
| 0 | `u8[8]` | magic `AICUKR1\0` |
| 8 | `u16` | schema major, 1 |
| 10 | `u16` | schema minor, 0 |
| 12 | `u32` | payload size, 128 |
| 16 | `u32` | exact CUDA architecture |
| 20 | `u32` | numeric kernel ID |
| 24 | `u32` | input dtype |
| 28 | `u32` | output dtype |
| 32 | `u32` | contiguous-layout flag |
| 36 | `u32` | grid X |
| 40 | `u32` | block X |
| 44 | `u32` | dynamic shared bytes |
| 48 | `u64` | element count |
| 56 | `u64` | exact CUBIN byte size |
| 64 | `u8[32]` | exact CUBIN SHA-256 |
| 96 | `u32` | input pointer alignment |
| 100 | `u32` | output pointer alignment |
| 104 | `u32` | module format, CUBIN |
| 108 | `u32` | launch ABI; pointer/pointer/numel for casts |
| 112 | `u8[16]` | reserved, zero |

The canonical scalar implementation uses a 256-thread block and a bounded
grid-stride loop. Grid X is exactly `min(ceil(numel / 256), 65535)`, shared
memory is zero, and tensor byte-size multiplication is checked. Unknown
kernel/dtype pairs, noncanonical launch geometry, invalid alignments, zero or
unknown module identity, PTX, and nonzero reserved fields fail closed.

## Receipt, capability, and native seam

An `AotCastValidationReceipt` binds the exact CUBIN identity, CUDA driver and
runtime versions, complete numeric problem set, exact symbol inventory, two
normal launches per problem, full-output reference comparison, bit-exact
normal repeat, and bit-exact CUDA Graph replay. Only a validated receipt may
create capabilities or commands.

Capabilities use provider ID 2 and ABI 1.0. Tensor shape remains part of the
exact `ProviderCapability`, while the kernel payload is shape-independent and
records dtype plus element count. Consequently, two shapes with the same dtype
and element count share one executable problem but retain distinct capability
records.

`AotCastCommand::Prepare` verifies the active architecture, module byte size and
SHA-256, tensor sizes and alignments, and numeric-ID symbol resolution. It
freezes caller-owned input/output addresses and launch arguments. `Execute`
only launches the resolved function on the caller stream; it does not load a
module, allocate, synchronize, search, or fall back.
