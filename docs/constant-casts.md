# Offline constant widening

Source compilation widens immutable BF16 constant inputs to F32 once during
weight packing. Native constant-only cast commands are removed and repeated
uses of one source weight share one derived constant. This is not a change to
GEMM precision: an F32/TF32 reference path remains F32/TF32. A future BF16 or FP8
executable form need not use this representation.

Conversion expands the little-endian BF16 representation into the high 16 bits
of F32, with zero low bits, without host floating-point arithmetic. Packing is
streamed with bounded buffers and supports split input elements across chunks.
The source dtype, shape and byte count are rechecked. Unreferenced original
weights are not packed; the derived F32 copy can increase resident model size.

Dynamic casts, mutable sources, partial accesses and exported cast outputs
remain runtime operations. Final liveness handles aliases; no persistent device
pointer is stored. The AIM build records removed commands, derived source IDs,
shared uses and bytes; offline verification checks this receipt against the plan.
The existing command ABI, numerical precision and runtime are unchanged.

CPU tests cover all 65,536 BF16 bit patterns, chunk splitting, shared packing,
exported/dynamic values and corrupt source metadata. On SM120/CUDA 12.8, a local
GPU probe also compared every bit pattern (including signed zero, subnormals,
infinities and NaNs) with the original `__bfloat162float` cast kernel under
poisoned, changed-input and restored CUDA Graph replay. All were bit-identical.
Synthetic bit patterns establish conversion correctness, not model fidelity or
performance; those require a separate real-input E2E comparison.
