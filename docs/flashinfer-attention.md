# FlashInfer attention provider (experimental)

AgInfer pins [FlashInfer](https://github.com/flashinfer-ai/flashinfer) as a Git
submodule for two currently delivered native attention regions. The dependency
is source-only at build time: deployment does not import FlashInfer's Python,
PyTorch, TVM FFI, or JIT layers.

The locked dependency identity is:

- FlashInfer release `v0.6.18`, source commit
  `69ff11fc4954396d98326656dc85debd2223f637`;
- FlashInfer's CCCL gitlink
  `876867684f7fac130e0f5911236e0a92a970d4fd`;
- FlashInfer's CUTLASS gitlink
  `b46b16d003484063bca4ed365e44095c4c6ed633` was used for the reproducible
  FA3 audit but is not consumed by the selected FA2 product build;
- Apache-2.0 upstream license and notice files remain in the submodule.

Initialize only the dependencies consumed by this provider:

```bash
git submodule update --init third_party/flashinfer
git -C third_party/flashinfer submodule update --init 3rdparty/cccl
```

Other FlashInfer submodules are not required by this implementation.

## Exact regions

The denoise variant accepts only SM120 and this BF16 GQA boundary:

```text
query       [1, 8, 50, 256]       BHQD BF16
key/value   [1, 1, 1018, 256]     BHQD BF16
mask_2d     [1, 50, 1018]         dense BOOL
output      [1, 50, 8, 256]       BSHD BF16
scale       0.0625
```

One fixed FlashInfer FA2 kernel reads the pre-broadcast 2-D mask and writes
BSHD directly. Lowering therefore fuses the mask head-broadcast, attention,
and post-attention transpose; the following row-major reshape remains a
memory-plan alias. The command is rejected unless every fused edge and tensor
signature matches exactly.

`Prepare` validates the 192-byte payload, target architecture, fixed source
commits, tensor sizes, alignment, and non-aliasing output, then opts into the
fixed 49,152-byte dynamic shared-memory launch. `Execute` constructs fixed
parameters and performs one kernel launch on the caller-owned stream. It does
not allocate, synchronize, inspect the device, select a tactic, or fall back.

The prefix variant accepts only SM120 and this second BF16 GQA boundary:

```text
query       [1, 8, 968, 256]      BHQD BF16
key/value   [1, 1, 968, 256]      BHQD BF16
pad_mask    [1, 968]              BOOL
output      [1, 968, 8, 256]      BSHD BF16
scale       0.0625
```

Its command reads the shared pad mask directly and fuses the key broadcast,
query broadcast, logical-and, head broadcast, attention, and exclusive output
transpose. For a padded query, the semantic graph assigns one equal finite
logit to every key and therefore returns `mean(value)`, not zero. The kernel
uses an all-zero logit row for that case, which is equivalent by softmax's
per-row shift invariance. Valid queries retain the finite BF16-minimum key
mask. Lowering refuses the region unless all four shared mask operations and
every attention/transpose edge have the exact, exclusive dataflow.

`Prepare` selects only the numeric variant encoded in the fixed payload; it
does not select a tactic or fallback implementation. Both variants launch one
fixed FA2 kernel from `Execute` on the caller-owned stream. The CUDA translation
unit is compiled for real `sm_120a` code only. Embedded PTX is an acceptance
failure.

## FlashAttention-3 audit

The pinned FlashInfer FA3/Hopper single-prefill entry was instantiated with its
fixed CUTLASS source for BF16, head dimension 256, and `sm_120a`. It is not a
valid implementation for the prefix region:

- `MaskMode::kCustom` returns `cudaErrorNotSupported` at the public entry;
- the unmasked head-dimension-256 instantiation fails the SM90 GMMA selector on
  `sm_120a` with `No eligible GMMA operator for request configuration`.

FA3 is therefore audited, not silently substituted. Reproducing that audit
requires initializing `3rdparty/cutlass`; normal FA2 builds require only CCCL.

## Deliberate exclusions

- F32 vision attention with head dimension 72 is outside this FlashInfer FA2
  variant.
- Other sequence lengths, head dimensions, mask graphs, dtypes, and target
  architectures are not inferred from these two receipts.

Unsupported regions stay unresolved at compile time. They never silently
route through this command.
