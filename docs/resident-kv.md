# Resident prefix/suffix KV storage

`--resident-kv` is an opt-in compiler transformation, independent of
`--fuse-projections`. Supply it to both `select-algorithms` and `compile`.
Selection receipts bind the transformed program identity. No runtime API or
Python inference dependency is added.

The source pattern is a prefix state written before inference steps, with an
exclusive state-read feeding `concat(prefix, current)` at each step. The
transformation allocates a full prefix-plus-suffix state and emits explicit
`state_update` operations:

1. Write the prefix region once per entry invocation.
2. Overwrite the suffix region for each step, without appending or advancing it.
3. Read the full state directly in attention.

The initial matching envelope is singleton batch/KV-head, BF16 CUDA state with
static shape. It requires one prefix refresh before all uses in the expanded
entry schedule, consistent suffix capacity, exclusive concat consumption by a
supported non-view operation, and no nested calls in the suffix reader.
Unsupported or incomplete lowering fails explicitly. No matching uses model
IDs or checkpoint parameter names. Build manifests record the transformed state
shapes under `resident_kv` and retain the source program digest.

The generic IR `state_update` has a CPU reference implementation. The initial
native form pairs two contiguous BF16 segment writes with equal bounds into
one vectorized copy kernel. Other ranks/sliced dimensions remain valid IR but
require a delivered native form; they do not silently fall back. The payload
is 128-byte little-endian `AISTU1`, ABI 1.0, with SM120, total/count/offset in
elements, CUBIN size/SHA-256 and zero reserved bytes. Prepare checks vector
alignment, ranges, access and source/destination or destination/destination
overlap. Execute only launches the fixed kernel on the caller stream.

For the current PI0.5 profile, each of the 18 layers has K and V states shaped
`[1,1,1018,256]`. Slots `[0:968]` hold the observation prefix and `[968:1018]`
hold the current 50 action tokens. RoPE positions are unchanged and are not
cache slot indices. Every new observation refreshes the prefix; every denoise
step recomputes the action suffix. There is no hidden initialized flag and no
input-dependent cache fill during Prepare.

This removes 180 repeated prefix copies, but retains 18 prefix-write and 180
suffix-write launches. Full state storage is 18,763,776 bytes versus 17,842,176
bytes originally. It eliminates about 170.16 MiB of logical copied prefix
payload per inference, not necessarily that much DRAM traffic. Initial short
paired E2E timing did **not** demonstrate a latency win; the option stays off
by default. This is a storage/data-movement optimization and a foundation for
future RoPE/store fusion, not a proven end-to-end speedup.

Validation includes CPU source/transformed parity, bounded-slice refusals,
state refresh, suffix overwrite, non-escaping views, payload validation, and an
optional GPU poison/partial-write/capture/overlap test. Real-model validation
must check changed observations, repeat/restore equality, independent sessions,
unchanged numerical gates, exact command counts and zero fallback. Heavy
fixtures, AIMs and timing reports stay outside Git. A native kernel capture
test does not certify model CUDA Graph replay.

Design references are the [vLLM slot-based cache writes](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu)
and [SGLang fused RoPE/store](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/elementwise/rope.cuh).
No implementation is copied and neither host becomes a dependency. Full paged
allocation and RoPE fusion are not part of this change.
