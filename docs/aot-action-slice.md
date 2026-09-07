# Exact terminal action-slice AOT command

AgInfer provides one fail-closed SM120 command for the tensor-only PI0.5 output
boundary

```text
padded normalized action F32 [1,50,32]
    -> normalized action F32 [1,50,7]
```

The operation keeps columns `[0,7)` of every 32-wide row. It is not a
contiguous prefix of the input buffer and cannot be represented by AgInfer's
row-major alias rules. `aginfer_action_slice_f32_r50_w32_o7` performs the
required strided reads in one caller-stream launch without allocation,
synchronization, module loading, tactic selection or fallback.

The exact capability matches the opcode, input/output dtype, shape, layout,
axis and bounds. The command lowerer additionally requires the result to be a
terminal entry output with no consumer. Selection does not inspect a model ID,
function name or debug value name. The fixed 192-byte payload binds the
variant, launch geometry, byte bounds, alignment, target architecture and
caller-owned CUBIN size/SHA-256. `Prepare` rejects mismatched modules or
architectures, null, short or misaligned bindings, and any input/output
overlap.

Public tests compare every row with the ProgramIR reference, prove that a
contiguous-prefix interpretation is wrong, cover payload corruption, exact
matching, terminal-output refusal, receipt gates and deterministic lowering.
Target-device acceptance reads the frozen E0 padded and seven-dimensional
normalized action tensors, requires all 350 outputs, repeat and CUDA Graph
replay to be bit exact, zero racecheck hazards, bounds/alias/architecture/module
negatives, and a PTX-free byte-reproducible CUBIN. Paired timing against an
independent row-slice kernel is recorded as diagnostic evidence only; this
terminal memory copy makes no standalone speedup claim. Model-level fidelity
and latency remain end-to-end acceptance gates.
