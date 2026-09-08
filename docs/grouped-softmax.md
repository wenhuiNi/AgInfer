# Grouped BF16 softmax

`select-algorithms --grouped-softmax` selects four warps per CTA only for the
small-row BF16 materialized attention form (Q50/K1018). Compilation consumes
the resulting fixed algorithm file without an additional flag. Prefix Q968
and F32 vision retain the original single-warp form: prefix's grouped candidate
was numerically correct but regressed in the local kernel experiment.

One warp still owns one row. Lane-to-key mapping, finite mask fill, BF16 score
scaling and probability rounding, exp and XOR-shuffle reduction order are
unchanged. Four independent rows now share a CTA; no cross-warp reduction or
shared-memory communication is introduced. QK/PV algorithms, score workspace,
GQA layout and output layout remain unchanged. This is not FlashAttention or
an online-softmax approximation.

The 192-byte `AIRAT3` payload explicitly selects the four-warp form (major3,
minor0); its other fields match `AIRAT2`. It is legal only for library-backed
BF16 variants. Old `AIRAT1`/`AIRAT2` payloads, kernel symbols and launch shapes
remain supported; changing the launch shape of an old AIM's embedded kernel
would not be safe. Prepare resolves the selected symbol and fixed launch once.
There is no runtime tuning or fallback. The low-level Q968 grouped form remains
available for experiments but is not selected by the public switch.

CPU gates cover canonical parsing, illegal vision/legacy combinations and
algorithm-report compatibility. Local GPU gates compare old/new outputs on
both lengths, pad-mask holes, fully masked rows, mask changes/restoration,
Graph replay and a deliberately ignored-mask negative control. The timing
fixture uses eight independent original-score buffers per Graph; upload and
download are outside timing. Synthetic scores are only a local kernel test.
Real-input full-model parity and paired E2E timing remain separate gates;
small local improvements do not establish an E2E gain.
