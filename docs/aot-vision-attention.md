# AOT CUDA vision attention provider v1

The built-in AOT CUDA provider implements one exact F32 vision-attention
region that is not covered by the pinned FlashInfer FA2 variant. It does not
claim a generic attention envelope.

## Exact region

The semantic ProgramIR attention site is:

```text
query/key/value  [1, 16, 256, 72]    BHSd F32
mask             [1, 16, 256, 256]   dense BOOL
output           [1, 16, 256, 72]    BHSd F32
kv_group_size    1
scale            1 / sqrt(72)
mask_fill        dtype_min
```

The executable region is larger and requires an exact producer/consumer
chain. Each query, key, and value must come from an exclusive
`BSHD -> BHSd` transpose. The mask must come from a shared broadcast of one
BOOL value. The attention output must have one `BHSd -> BSHD` transpose
consumer. The command consumes the pre-transpose BSHD tensors and the
pre-broadcast mask value, then writes BSHD directly. This fuses three input
transposes, the mask broadcast, attention, and the output transpose.

The mask value used by the current model is true. The kernel also preserves
the declared finite `dtype_min` behavior for a false value: all logits are
equal and the result is the uniform mean of value tokens, rather than zero or
NaN.

## Kernel and command contract

`kernels/vision_attention.cu` exports one fixed symbol:

```text
aginfer_vision_attention_f32_bshd
```

The launch uses grid `(256, 16, 1)` and block `(256, 1, 1)`, with one block
per query token and head. The same target-specific, PTX-free CUBIN also holds
the cast and pointwise kernels.

The 192-byte numeric payload binds the exact tensor dimensions, dtypes,
layouts, scale, mask semantics, launch geometry, CUBIN byte size, and CUBIN
SHA-256. It contains no pointers or symbol strings.

`AotVisionAttentionCommand::Prepare` validates the payload, active SM120
target, caller-owned module identity, tensor sizes, alignment, output
non-aliasing, and fixed symbol. `Execute` performs one driver launch on the
caller stream. It does not allocate, synchronize, load modules, inspect the
device, choose a tactic, or fall back.

Unsupported shapes, dtypes, layouts, mask producers, extra consumers, and
architectures remain unresolved at compile time.
