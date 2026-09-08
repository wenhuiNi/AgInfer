# Packed QKV, RoPE and KV layout fusion

The native compiler joins four existing commands into one SM120 AOT launch:
packed projection split, query RoPE, key RoPE, and attention KV packing. This is
a physical layout fusion, not a new precision policy or an attention algorithm.
It runs automatically when those exact forms are connected; other shapes and
the separate resident-cache experiment keep their existing path.

## Boundary and semantics

The fixed `AIQRP1` payload uses the existing 128-byte projection descriptor
layout, but a separate magic and stricter dimensions. It contains no pointers.
The C++ parser accepts only rows 50 and widths `(2048, 256, 256)` for this form.

| Operand | Access | Type and physical layout |
|---|---|---|
| Packed QKV | Read | BF16 `[1,50,2560]`, per row Q then K then V |
| Positions | Read | I32 `[1,50]`, supplied on every replay |
| Prefix K / V | Read / read | BF16 `[1,1,968,256]` each |
| Rotated Q | Write | BF16 `[1,8,50,256]` |
| Packed K / V | Write / write | BF16 `[1,1,1018,256]` each |

The kernel reads Q/K directly from the packed projection. It reuses the existing
RoPE device function with a compile-time input row stride: BF16 inverse
frequencies, BF16 sin/cos, two BF16 products and a BF16 sum/subtraction are
preserved. It does not replace this arithmetic with an F32 FMA.

Disjoint block ranges write query rotation, suffix key rotation, prefix copies,
and suffix values. All output elements are overwritten. Prefix tensors remain
read-only and are still copied into the packed output: this optimization does
**not** remove repeated prefix-copy traffic. Observation-cadence cache filling,
camera count, token lengths, masks, attention scaling and denoising steps are
unchanged. Prepare binds one fixed symbol; Enqueue makes one launch, with no
allocation, tactic search, synchronization, or cache pointer reassignment.

## Compiler and binding guards

The pass joins already-validated provider boundaries through memory-plan alias
roots. Split Q/K/V and the rotated intermediate key must be private arena
storage with exactly the expected physical reader. Public outputs, including
reshape aliases, and shared intermediates cannot disappear. Query and key must
read the same position storage; partial operands, differing module identities,
wrong shapes and mutable positions are not fused. The normal command-ordering
mechanism retains state-read receipts and checks state dependencies.

Coverage is transferred through `placements_from_partial_lowering`, including
any shared producers. Unused intermediate allocation disappears in the existing
command-driven liveness pass. Capability registration remains a candidate, not a
manufactured numerical or capture receipt.

C++ Prepare validates the exact operand count, four-read/three-write access,
byte spans, 16-byte alignment, output overlaps, SM120 and CUBIN identity. Read-only
aliases remain valid, matching the original KV pack contract. Old split
payloads and symbols remain supported. Runtime support for the new magic is
required to load a newly fused AIM; older runtimes must refuse it.

## Validation

Public CPU tests check alias-based matching, coverage conservation, eliminated
intermediate allocation, pass idempotence, and refusal of exported/shared,
partial, different-position, stale-module and resident-cache forms. The payload
parser is also tested in a no-CUDA build.

The small GPU contract test compares all three outputs against the original
four-launch chain. It checks eager execution, a one-node captured graph,
poisoned outputs, changed QKV/positions/prefix, restoration, unchanged inputs,
and span/access/alignment/alias/module/architecture refusals. Synthetic inputs
in this contract test are not calibration or model fidelity evidence.

### PI0.5 E2E-A result (2026-09-08)

RTX 5090 / SM120, CUDA 12.8 and cuBLASLt 120803; resident preprocessed inputs to
normalized actions, three physical cameras, prefix 968, suffix 50 and ten
denoising steps. This is not raw preprocessing or a closed-loop task result.
Input provenance is the existing real episode observation through the source
processor, with fixed token/mask/noise assets. No synthetic calibration was used.
The baseline is the accepted residual/norm-fused, reference-precision artifact;
the opt-in BF16-large-linear experiment was **not** enabled. All linear/patch
algorithm payloads are identical, and attention differs only in CUBIN identity.

| Measurement | Baseline | Fused |
|---|---:|---:|
| Commands | 3,549 | 3,009 |
| CUDA Graph nodes | 4,955 | 4,415 |
| QKV/RoPE/KV groups, each inference | 180 × four launches | 180 × one launch |
| Pair 1, AB (ms) | 50.311346 | 50.735923 |
| Pair 2, BA (ms) | 51.184792 | 49.695405 |
| Pair 3, AB (ms) | 50.308086 | 49.687964 |
| Short-run median (ms) | 50.311346 | 49.695405 |

The median decreased by approximately 1.22%, but one pair regressed. This short
run does **not** establish stable latency improvement or deployment P50/P95.
Timing is host wall time through stream synchronization, with output downloads
outside the timing interval; both arms use captured graphs. Different earlier
timing windows must not be subtracted to claim cumulative acceleration.

Original, jointly changed image/noise, and restored inputs all produced bit-exact
baseline/candidate actions. The unchanged official-action gate passed:
cosine `0.99999635435416`, maximum absolute error `0.0039391219615936`, linear
p99 error `0.0030150011181831`. These differences are relative to the source
oracle, not between the two native arms. Provider ledgers report zero fallback;
graph launches are five baseline and six candidate, with the expected node drop.

Three vision patch commands, 18 prefix KV stores, 1,350 linears, 278 attentions,
and input/output port contracts remain unchanged. Arena `114278400`, state
`17842176`, workspace `14992384`, and weight `7086076160` byte totals are also
unchanged. Eliminated intermediates remove approximately 101.38 MB of logical
read/write traffic per inference, not a measured DRAM saving; peak arena storage
is dominated elsewhere.

The candidate has 180 `AIQRP1` commands, no standalone projection splits or KV
packs, and 35 remaining prefix RoPE commands. The artifact is 7,090,134,464 bytes,
file SHA-256 `2bf49b66c89322e1108e5b912b28a73ff450191ea37254e6de7342b84b01ba25`,
build SHA-256 `8e2e86eb8c59d7c9ba4f63bad649547eb896a7c9e57108849a9bf2b883a1e875`.
The model, binaries, inputs, and paired-run harness remain outside Git. Full
offline verification passed after GPU timing, never concurrently.

Acceptance includes 288 Python tests, 31 non-GPU CTests, the no-CUDA payload
test and one real short E2E run. Local GPU validation ran twice: the initial
kernel/capture check and a short binding regression after review found that
read-only prefix aliases should remain legal. The latter changes only host
validation, not the kernel, payload or model computation; model E2E was not
repeated. Further priorities
are current hot-path profiling and full attention fusion, followed by gated
mixed-precision work; camera/token pruning remains excluded.
