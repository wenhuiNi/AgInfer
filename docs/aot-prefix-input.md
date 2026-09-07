# Exact prefix-input assembly AOT command

AgInfer provides one fail-closed SM120 command for the fixed prefix assembly
boundary

```text
3 x projected image F32 [1,256,2048]
embedding BF16 [257152,2048] + token IDs I32 [1,200]
3 x image-valid BOOL [1] + token-valid BOOL [1,200]
    -> prefix F32 [1,968,2048]
    -> pad mask BOOL [1,968]
    -> position IDs I32 [1,968]
```

The first 768 prefix rows are copied from the three projected images. The last
200 rows gather the tied BF16 language embedding, multiply by the BF16-rounded
`sqrt(2048)` value (`45.25`), round the product to BF16, and then convert it to
F32. This ordering is part of the model semantics; multiplying in F32 without
the intervening BF16 rounding is not equivalent. The pad mask expands each
image-valid byte to 256 entries and appends the token mask. Position IDs are
the inclusive sum of that exact, possibly non-contiguous mask minus one.

The structural lowerer replaces 13 operations: language gather, literal-scale
broadcast, multiply and cast; the embedding concat; three runtime image-mask
broadcasts and the pad concat; and the pad cast, cumulative sum, literal
negative-one broadcast and add. It requires exact tensor contracts, literals,
attributes, invocation identity and exclusive intermediate consumers. The
three outputs remain independently visible. Selection does not inspect a model
ID, function name, call number, debug value name or token contents. Any changed
shape, dtype, layout, constant, dataflow, consumer set or target fails closed.

`aginfer_prefix_input_f32_bf16_s968_d2048` performs one caller-stream launch.
It does not allocate, synchronize, load a module, choose a tactic or invoke a
fallback. An out-of-range token is guarded before the embedding read and emits
a canonical quiet NaN row. This prevents an out-of-bounds device access and
makes invalid data observable, but is not an enqueue-time error: a future
complete runtime must validate dynamic token IDs before launch if its public
contract promises synchronous input rejection.

The fixed 192-byte little-endian payload records the exact variant, launch
geometry, alignment and byte bounds plus the caller-owned CUBIN size/SHA-256.
`Prepare` validates the architecture, module identity, symbol, every binding,
size and alignment, all read/write overlaps and all output/output overlaps.

Public tests cover Python/C++ payload agreement, strict receipt gates,
deterministic structural fusion, literal/dataflow/consumer/public-output/target
negatives and exclusion of the fused literals from offline materialization.
Target-device acceptance uses the frozen E0 inputs and oracle: all 1,982,464
prefix values, 968 mask bytes and 968 positions must be bit exact; the oracle
must independently match the stated BF16 reference; repeat and CUDA Graph
replay must be bit exact; mask-hole, BF16-rounding and invalid-token controls
must fire; racecheck must report zero hazards; bounds, alias, architecture and
module-identity negatives must pass; the CUBIN must be PTX-free and
byte-identical across independent builds; and paired alternating timing must
beat the original 13-launch chain. This is a provider-region contract, not an
end-to-end model-support claim.
