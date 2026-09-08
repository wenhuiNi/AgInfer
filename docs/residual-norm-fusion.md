# Rounded residual to adaptive RMSNorm fusion

With the existing compile-only `--fuse-residual` policy, native lowering can
combine a compact gated residual with its following no-gate adaptive RMSNorm.
The new `AIGRN1` form has four inputs (activation, previous modulation, residual,
next modulation) and two outputs (updated residual, normalized activation).
Both outputs remain available to other consumers; the residual is not removed.

The kernel preserves gate-to-BF16, product-to-BF16 and residual-to-BF16 rounding
before normalization. Each thread retains its four updated residual elements
while executing the original 256-thread reduction order. This avoids another
launch and rereading the residual for normalization. It does not fuse a GEMM or
change model precision, mask, token count or camera profile.

Matching joins already-validated physical forms through their shared value, not
through model/layer names. Mutable or aliased modulation is conservatively
excluded. Shared source operations are assigned using the existing coverage
mechanism before moving a norm to another provider group; final ordering and
liveness still validate all dependencies. Old `AIANG1`/`AIGRD1` symbols and
launch shapes retain their meanings. New binding checks include both output
spans, read/write access and pairwise output non-overlap.

Public CPU tests cover two-output dependency/coverage and retained negative
paths. GPU tests compare both results bit-for-bit against the separate kernels
in eager and poisoned changed/restored Graph runs, with binding refusals and
rounding negative controls. The local model exercised 350 fused commands with
zero fallback and bit-identical original/changed/restored actions. Short paired
E2E timing was mixed (two improvements, one slight regression), so no stable
latency improvement is established by this change.
