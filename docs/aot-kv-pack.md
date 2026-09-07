# Exact denoise KV-pack AOT command

AgInfer provides one fail-closed SM120 command for the PI0.5-style denoise
attention boundary

```text
prefix_k BF16 [1,1,968,256] + current_k BF16 [1,1,50,256]
prefix_v BF16 [1,1,968,256] + current_v BF16 [1,50,1,256]
                                      -> packed_k/v BF16 [1,1,1018,256]
```

The kernel copies both K and V with 16-byte vectors in one caller-stream
launch. Because this exact variant has one KV head, current V in BSHD and BHSD
order has identical physical bytes; the command therefore absorbs the
singleton transpose without moving those bytes separately. It replaces two
exclusive concat operations and that transpose. The packed tensors remain the
ordinary contiguous inputs of the separately delivered FlashInfer attention
command.

The structural lowerer anchors on the two exact concat signatures and verifies
the adjacent GQA attention boundary, same invocation, state-backed prefix
aliases, current K RoPE producer, exclusive singleton V transpose, exclusive
consumers, and absence of public fused intermediates. It uses no model ID,
layer number, or debug-name match. A changed shape, dtype, layout, concat axis,
mask/attention contract, extra consumer, visible intermediate, or overlapping
region causes fail-closed refusal.

The 192-byte little-endian payload records the exact target, layout tags,
shapes, launch geometry, alignment, all six byte bounds, and CUBIN
size/SHA-256. `Prepare` checks the loaded module identity and rejects null,
misaligned, short, or partially overlapping bindings. `Execute` resolves to
`aginfer_kv_pack_bf16_h1_s968_s50_d256`; it performs one launch and does not
allocate, load a module, synchronize, select a tactic, or invoke a fallback.

Pinned FlashInfer's paged-prefill API can represent a custom mask, but it
requires one paged backing buffer plus planning metadata and does not consume
the existing prefix/current K/V pointer segments directly. The first tensor-only
BF16 E2 baseline therefore uses this explicit, observable pack command. Direct
provider-managed placement into persistent contiguous KV storage is a later
post-E2 optimization; it must not be introduced as hidden runtime state.

The public tests cover exact payload parsing, strict receipt gates, structural
recognition, deterministic lowering, and visible-intermediate/wrong-target
refusal. The target-device gate additionally requires complete bit-exact K/V
comparison, bit-exact repeat and CUDA Graph replay, zero racecheck hazards,
architecture/bounds/partial-alias/module-identity negatives, a PTX-free CUBIN,
and byte-identical independent builds. This is a provider-region contract, not
an end-to-end model-support claim.
