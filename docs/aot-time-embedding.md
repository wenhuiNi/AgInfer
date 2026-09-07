# Exact time sinusoidal-embedding AOT command

AgInfer provides one fail-closed SM120 command for the fixed PI0.5/OpenPI
sinusoidal boundary

```text
timestep F32 [1]
dimension=1024, min_period=0.004, max_period=4.0
    -> embedding F32 [1,1024]
```

The 512 periods form a geometric sequence including both endpoints. Periods,
scales, angles, sine and cosine are evaluated in Float64, the 512 sine values
are concatenated before the 512 cosine values, and only then is the result
converted to F32. This matches the source implementation's CUDA semantics;
using an F32 period grid is detectably different.

`aginfer_time_embedding_f32_d1024` performs one caller-stream launch. It does
not allocate, synchronize, load a module, select a tactic or invoke a fallback.
The exact capability matches opcode, input/output dtype, shape, layout, period
attributes and target architecture. It does not inspect a model ID, function
name, denoise-step number, debug value name or timestep contents.

The fixed 192-byte little-endian payload binds the variant, F64 period
semantics, sin-then-cos layout, launch geometry, byte bounds, alignment, target
architecture and caller-owned CUBIN size/SHA-256. `Prepare` resolves the symbol
and rejects a mismatched architecture or module, null, short or misaligned
bindings, and any input/output overlap.

Public tests cover the geometric endpoints and ordering, exact capability and
payload agreement, corrupt fields, receipt gates, deterministic lowering and
stage identity. Target-device acceptance compares all 10,240 outputs for the
ten E0 timesteps with an independently generated LeRobot/OpenPI CUDA oracle,
requires bit-exact repeat and CUDA Graph replay, detects F32-period semantics,
reports zero racecheck hazards, exercises bounds/alias/architecture/module
negatives, requires a PTX-free byte-reproducible CUBIN, and uses paired timing
against an independent three-launch semantic CUDA chain. This is a provider
region contract, not an end-to-end model-support claim.
