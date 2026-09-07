# Exact denoise suffix-metadata AOT command

AgInfer provides one fail-closed SM120 command for the denoise metadata
boundary

```text
prefix_pad BOOL [1,968]
    -> attention_mask BOOL [1,50,1018]
    -> positions I32 [1,50]
```

For each of the 50 suffix queries, the first 968 mask entries preserve the
prefix pad mask byte for byte and the final 50 entries are true. Positions are
`sum(prefix_pad) + arange(50)`. The implementation therefore preserves holes
in a non-contiguous prefix mask; it does not infer a valid-token interval from
the count.

The structural lowerer closes the exact broadcast/concat/cast/reduce/cumulative
sum/add region. It requires literal `true` and `-1`, exclusive intermediate
consumers in one invocation, the exact denoise head-mask consumer, only
split-half RoPE consumers for the resulting positions, and no public
intermediate. Selection uses no model ID, denoise-step number, function name,
or debug value name. A changed dtype, shape, layout, constant, attribute,
consumer set, or target architecture causes fail-closed refusal.

`aginfer_suffix_metadata_bool_s968_s50` uses one caller-stream launch. Its 199
blocks write the full mask in parallel; block zero also reduces the prefix pad
mask and writes the positions. The command does not allocate, synchronize,
load a module, select a tactic, or invoke a fallback.

The fixed 192-byte little-endian payload records the exact tensor/layout
variant, launch geometry, alignment, buffer bounds, and caller-owned CUBIN
size/SHA-256. `Prepare` validates architecture, module identity, symbol,
bindings, sizes, alignment, and every pairwise overlap before constructing an
executable command.

Public tests cover Python/C++ payload agreement, strict receipt gates,
deterministic 12-op region lowering, constant and consumer mutations, public
intermediate rejection, and target mismatch. Target-device acceptance also
requires complete outputs against the real E0 558-valid mask, a hole-mask
negative control, bit-exact repeat and CUDA Graph replay, zero racecheck
hazards, bounds/alias/module negatives, a PTX-free CUBIN, byte-identical
independent builds, and paired timing against the original 12-launch chain.
This is a provider-region contract, not an end-to-end model-support claim.
