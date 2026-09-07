# Exact prefix KV state-store AOT command

AgInfer provides one fail-closed SM120 command for the prefix cache-write
boundary

```text
key_after_rope BF16 [1,1,968,256] + value BSHD BF16 [1,968,1,256]
                         -> persistent key/value BF16 [1,1,968,256]
```

The command copies K and V with 16-byte vectors into two explicit schedule
state allocations in one caller-stream launch. With one KV head, the BSHD and
BHSD value tensors have identical row-major physical order. The memory planner
proves such singleton-axis transposes as aliases; no layout kernel or hidden
host copy is emitted. This identity-view rule is based on shape, permutation,
and row-major physical axis order and is not specific to a model or state name.

The structural lowerer distinguishes the V path from the same-shaped K view by
its dataflow. It requires one exact prefix GQA attention consumer, paired
adjacent K/V state writes in the same invocation, a split-half RoPE K producer,
the exact singleton V view, two distinct persistent state targets, and no
public intermediate. Selection uses no model ID, layer number, debug name, or
state-name pattern. A changed dtype/shape/layout/attention/RoPE contract,
consumer set, state target, or view proof causes fail-closed refusal.

The 192-byte little-endian payload records the target, layout tags, exact
shape, launch geometry, alignment, four buffer bounds, and CUBIN
size/SHA-256. `Prepare` resolves
`aginfer_prefix_kv_store_bf16_h1_s968_d256`, validates the caller-owned module
and all bindings, and rejects null, misaligned, short, or partially overlapping
ranges. `Execute` performs one launch and does not allocate, load a module,
synchronize, select a tactic, or invoke a fallback.

The public tests cover singleton-view planning, exact payload parsing, strict
receipt gates, structural pairing, deterministic lowering, visible-value and
duplicate-state refusal, and Python/C++ payload agreement. The target-device
gate additionally requires complete bit-exact state comparison, bit-exact
repeat and CUDA Graph replay, zero racecheck hazards,
architecture/bounds/partial-alias/state-alias/module-identity negatives, a
PTX-free CUBIN, and byte-identical independent builds. This is a
provider-region contract, not an end-to-end model-support claim.
