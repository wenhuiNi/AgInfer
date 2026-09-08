# Direct modulation gate consumption

`compile --fuse-residual` also negotiates a paired physical form between exact
adaptive RMSNorm and rounded residual lowering. The BF16 `[1,50,1024]` gate
must have one multiplication consumer, no export or view/shared consumer, and
the modulation must be a non-aliased immutable SSA value (not mutable state).
Unsupported boundaries keep the existing providers.

ProgramIR still expresses broadcast/cast/multiply/add semantics. In the physical
command stream, `AIANG1` computes normalized hidden only; `AIGRD1` reads the
original F32 `[1,3072]` modulation and consumes its last 1024 elements directly.
No command reads or writes the broadcast gate, so existing command liveness
removes its allocation. No undeclared short tensor or cross-observation cache
is introduced. Modulation remains live until residual consumption.

The residual kernel explicitly performs F32 gate -> BF16, F32 product -> BF16,
then F32 addition -> BF16. Removing either conversion is a different numerical
operation, not an accepted fusion. The norm reduction order is unchanged. The
old norm and rounded-multiply-add symbols/payloads remain available to old AIMs.

Both new payloads are 128 bytes, with distinct magic, ABI1.0, SM120, fixed numel
51200, exact module byte count/digest and zero reserved fields. They reuse the
rounded-multiply-add wire field layout, not its kernel or operand contract.
Norm binds hidden/modulation/output (`rrw`); residual binds activation/modulation/
residual/output (`rrrw`). Prepare checks spans, alignment, access, output overlap,
target and module. Enqueue only launches fixed AOT kernels on the caller stream.

Acceptance covers public CPU pairing/refusal and gate liveness, both parsers
without CUDA, GPU eager and poisoned Graph parity with the old kernels, changed
modulation/restoration, and negative controls for omitted gate/product rounding.
Model validation must additionally check real-input action gates, changed
observation controls, target-path call counts and zero fallback. Logical bytes
eliminated are not a measurement of DRAM traffic or an E2E speedup.
