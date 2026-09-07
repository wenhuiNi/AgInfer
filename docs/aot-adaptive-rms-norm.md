# Exact adaptive RMSNorm AOT command

AgInfer provides one fail-closed SM120 command for the recurring adaptive
normalization region

```text
hidden_bf16 -> f32 RMSNorm -> norm * (1 + scale) + shift -> normalized_bf16
modulation_f32[scale | shift | gate]                  -> gate_bf16
```

The delivered boundary is fixed to `hidden=[1,50,1024]` BF16,
`modulation=[1,3072]` F32, and two BF16 `[1,50,1024]` outputs. Modulation is
three contiguous 1024-element vectors in `scale`, `shift`, `gate` order. The
epsilon is `1e-6`; accumulation, reduction, scale, and shift arithmetic are
F32.

The structural lowerer recognizes dataflow, tensor signatures, attributes,
and exclusive consumers. It does not match model IDs or debug names. Each
accepted region replaces the input cast, RMSNorm, three slices, three
broadcasts, scale/add/multiply/shift operations, and two output casts with one
command. Shared scalar-one broadcasts are removed only when every consumer is
an accepted region. A visible intermediate, extra modulation consumer, changed
slice, changed epsilon, wrong dtype/shape/layout, unsupported architecture, or
overlapping region causes lowering to fail rather than partially fuse.

The 192-byte little-endian payload records the exact target, variant,
shape/dtype/layout, launch geometry, alignments, byte bounds, epsilon,
accumulation mode, and CUBIN size/SHA-256. `Prepare` resolves the numeric
variant to `aginfer_adaptive_rms_norm_bf16_f32_1024`, checks all four bindings
and prohibits output aliases. `Execute` performs one caller-stream launch and
does not allocate, load a module, synchronize, select a tactic, or invoke a
fallback.

The public tests cover region recognition, deterministic lowering, visible
intermediate refusal, receipt gates, and Python/C++ payload agreement. The
target-device gate additionally requires full comparison of both outputs,
bit-exact gate output, repeat and CUDA Graph equality, zero racecheck hazards,
negative architecture/bounds/alias/module-identity checks, a PTX-free CUBIN,
and byte-identical independent builds. This is a provider-region contract, not
an end-to-end model-support claim.
