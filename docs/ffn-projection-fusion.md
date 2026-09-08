# Packed gate/up projection fusion

Use `--fuse-ffn` on both `select-algorithms` and `compile`. It composes with
`--fuse-projections` (QKV) and `--constant-evaluator PATH`, and implies the
existing BF16 GELU/multiply lowering. Omitting it preserves the previous path.
Selection and compilation must use the same transformed ProgramIR and CUBIN.

The source rewrite recognizes exactly two shared-input BF16 CUDA linears with
positive-zero bias, constant weights, batch 1, 1..2048 rows and aligned output
widths. One output must feed tanh GELU, then multiplication with the other;
neither projection nor the activated intermediate may escape that exclusive
boundary. Gate/up widths must match. No model or weight-name matching is used.

Weights are concatenated offline in gate-then-up output-channel order using
the existing streamed derived-constant packer. One fixed cuBLASLt GEMM emits
`[1, rows, 2*width]`. ProgramIR retains ordinary slices/GELU/mul as semantics;
the provider covers all four with one packed activation command, so there are
no runtime split copies. The packed payload binds numel, width and exact module
identity; `AIGMU1` remains byte-compatible for separate input buffers. The
packed form accepts a doubled read buffer and a distinct output buffer, checks
the full input span for overlap, and preserves GELU's BF16 intermediate rounding.

New compilation uses `AIGMT1`: grid.y selects the first row and grid.x selects a
256-column tile, removing per-element division/modulo when locating packed
halves. A distinct kernel symbol preserves compatibility: existing `AIGMP1`
packages still launch their embedded one-dimensional kernel with its original
grid. For more than 128 rows, the tiled kernel strides over rows, with at most
4096 blocks (968 rows and width 16384 use a 64-by-64 grid). Every previous
<=128-row launch shape is preserved, including wide rows, because an old AIM
can embed a tiled kernel without the new row loop. Older runtimes reject the
expanded row envelope. Changing only the launch shape of an old embedded
kernel is not safe.
The >128-row tiled path resolves the distinct
`aginfer_gelu_tanh_mul_packed_prefill_bf16` symbol at Prepare. This prevents
an old CUBIN containing only the non-striding tiled entry from being silently
accepted with the new grid. Small rows retain the previous symbol.
Both forms cover odd-width tails and retain identical element arithmetic.
Fewer indexing instructions alone do not establish an E2E speedup; require
paired timing on the deployed shape and Graph policy.

BF16 prefix FFNs within this envelope can now use the same packed form;
QKV remains restricted to at most 128 rows. Nonzero bias, shared/exported
intermediates and unrelated paired linears stay unchanged. The compiler's exact native forms
still limit the whole-program envelope; the local kernel's broader shape
contract does not claim other model families have an end-to-end frontend.

Tests cover reversed projection declaration order, exclusive boundary refusal,
packing order, four-op coverage, 129/968/2048-row acceptance and 2049-row
refusal, old/new payload parsing, real prefix and denoise widths
and an odd-width tail, eager/Graph bit-exact comparison with the original
activation kernels and invalid packed buffer bindings. Numerical validation
of the merged GEMM still requires model E2E; local activation parity alone is
not evidence for changing the GEMM shape.

## Prefill extension acceptance (2026-09-08)

On RTX 5090 / SM120, CUDA 12.8 and cuBLASLt 12.8.3, the fixed PI0.5
profile retains three 224-by-224 images, 968 prefix tokens, 50 action tokens
and ten denoising steps. Native DCE leaves 17 live prefix FFNs: their 34
`(M,N,K)=(968,16384,2048)` gate/up GEMMs become 17
`(968,32768,2048)` GEMMs followed by packed activation. FLOPs and precision
are unchanged. The new shape selects the same algorithm/tile/stages as the
old shape; all 15 common linear payloads are unchanged. Patch and attention
payloads differ only in AOT module identity.

One real-fixture E2E-A comparison (resident device inputs to normalized actions)
used identical CUDA Graph policies, two warmups, three alternating timing
pairs, changed image/noise and candidate restore: 11 Enqueues total. Baseline /
candidate times in ms were 56.522762 / 54.884979, 56.491588 / 54.854559,
56.339234 / 54.774742. Medians were 56.491588 / 54.854559 (2.90% lower).
This is short-run evidence without locked clocks or GPU isolation, not a
stable percentile estimate. Do not compare it directly with earlier timing
windows or attribute all savings specifically to GEMM versus activation.

Original, changed and restored outputs were bit-exact to the previous native
path; the original official-reference gates passed (cosine 0.99999635435416,
max absolute error 0.0039391219615936, p99 error 0.0030150011181831).
The changed-input control is not a held-out task-quality evaluation.
Fallback count was zero. Commands decreased 3009 -> 2992, Graph nodes
4415 -> 4398, and linear commands 1350 -> 1333. Packed activations increased
180 -> 197, replacing all 17 live separate-buffer prefix activations.
Arena/state/workspace remain 114278400 / 17842176 / 14992384 bytes;
weights grow by 32768 bytes for the wider shared zero bias. Public port and
state contracts are unchanged; internal value IDs are allowed to change.

289 Python tests, 31 non-GPU CTests, a no-CUDA parser test and one local GPU
contract run passed. The GPU run covers both packed forms at real prefix
size, eager and poisoned Graph replay, preserving BF16 rounding bit-for-bit.
Full artifact verification ran only after timing and passed. Heavy model,
fixture, AIM and paired harness artifacts stay outside Git.

Final compatibility review added the distinct prefill symbol above, sharing
the same force-inlined arithmetic with the tiled entry. Timing above belongs
to the pre-guard candidate, not a remeasurement of the final module. A
bind-only old/new-module check verifies old small-row acceptance, old
large-row refusal and new small/large-row acceptance. The final artifact
passed a separate five-Enqueue original/changed/restored E2E correctness
check (bit-exact, zero fallback, 4398 Graph nodes) and full artifact
verification; no additional performance pairs were used.
