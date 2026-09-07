# Packed gate/up projection fusion

Use `--fuse-ffn` on both `select-algorithms` and `compile`. It composes with
`--fuse-projections` (QKV) and `--constant-evaluator PATH`, and implies the
existing BF16 GELU/multiply lowering. Omitting it preserves the previous path.
Selection and compilation must use the same transformed ProgramIR and CUBIN.

The source rewrite recognizes exactly two shared-input BF16 CUDA linears with
positive-zero bias, constant weights, batch 1, 1..128 rows and aligned output
widths. One output must feed tanh GELU, then multiplication with the other;
neither projection nor the activated intermediate may escape that exclusive
boundary. Gate/up widths must match. No model or weight-name matching is used.

Weights are concatenated offline in gate-then-up output-channel order using
the existing streamed derived-constant packer. One fixed cuBLASLt GEMM emits
`[1, rows, 2*width]`. ProgramIR retains ordinary slices/GELU/mul as semantics;
the provider covers all four with one packed activation command, so there are
no runtime split copies. The `AIGMP1` payload binds numel, width and exact module
identity; `AIGMU1` remains byte-compatible for separate input buffers. The
packed form accepts a doubled read buffer and a distinct output buffer, checks
the full input span for overlap, and preserves GELU's BF16 intermediate rounding.

Large-row prefix projections, nonzero bias, shared/exported intermediates and
unrelated paired linears stay unchanged. The compiler's exact native forms
still limit the whole-program envelope; the local kernel's broader shape
contract does not claim other model families have an end-to-end frontend.

Tests cover reversed projection declaration order, exclusive boundary refusal,
packing order, four-op coverage, old/new payload parsing, real denoise width
and an odd-width tail, eager/Graph bit-exact comparison with the original
activation kernels and invalid packed buffer bindings. Numerical validation
of the merged GEMM still requires model E2E; local activation parity alone is
not evidence for changing the GEMM shape.
