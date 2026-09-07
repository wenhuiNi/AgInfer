# AOT CUDA pointwise provider v1

The built-in AOT CUDA provider implements these contiguous ProgramIR
pointwise semantics:

- `f32`, `bf16`, and `i32` same-shape `add`;
- `f32` and `bf16` same-shape `mul`.

Broadcasting, mixed dtypes, in-place output, activation epilogues, and fused
forms are outside this capability. The compiler must select another exact
provider or fail closed for those cases.

The product kernels live in `kernels/pointwise.cu` and are compiled into the
same target-specific, PTX-free AOT CUBIN as the cast kernels. Their fixed symbol
subset is:

```text
aginfer_add_f32
aginfer_add_bf16
aginfer_add_i32
aginfer_mul_f32
aginfer_mul_bf16
```

## Command contract

Pointwise commands use the shared 128-byte numeric CUDA-kernel payload
documented in `aot-cuda-cast.md`. Kernel IDs 4 through 8 map to the symbols
above and require the binary launch ABI:

```text
(const T* lhs, const T* rhs, T* output, uint64_t numel)
```

The canonical launch uses 256 threads and
`min(ceil(numel / 256), 65535)` blocks with a grid-stride loop and no dynamic
shared memory. The payload binds the exact target architecture, CUBIN byte
size and SHA-256, dtype, element count, launch geometry, and pointer
alignments. It stores no symbol strings or device addresses.

`AotPointwiseCommand::Prepare` verifies the payload, active architecture,
caller-owned module identity, all three tensor sizes and alignments, numeric
kernel ID, launch ABI, and resolved symbol. It then freezes caller-owned
addresses and launch arguments. `Execute` performs exactly one driver launch
on the caller stream; it does not allocate, load modules, synchronize, search,
or fall back.

## Validation receipt

Capabilities and lowering commands can only be created from an
`AotPointwiseValidationReceipt`. A receipt identifies the complete exact
problem set and requires, for every problem:

- a full-output reference comparison;
- two normal launches with bit-exact repeat output;
- bit-exact CUDA Graph capture/replay;
- an exact five-symbol pointwise inventory;
- a target-specific CUBIN with no embedded PTX.

Capabilities use provider ID 2 and ABI 1.0. Exact tensor shape remains part of
capability matching even when two shapes share the same dtype and element
count. Partial lowering records every emitted execution index and every
remaining execution index; it cannot silently claim whole-program coverage.
