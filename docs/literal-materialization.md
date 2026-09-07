# Compile-time literal materialization

`aginfer.lowering.build_literal_materialization(schedule, inventory)` removes
runtime expansion of static literal broadcasts. It is an offline compiler
stage: it emits immutable bytes and receipts, not a provider capability or a
runtime command.

The v1 stage considers only a required `broadcast_in_dim` whose input is a
schedule constant with a `literal:<site>` identity. The source constant must be
recoverable from the verified lowering inventory, and the output must have a
fully specialized CUDA row-major type using BOOL, I32, F32, or BF16. Runtime
inputs, model `constant_ref` values, symbolic shapes, unsupported dtypes, and
malformed axis mappings are not materialized. Execution indices already
claimed by fused provider regions must be passed through
`excluded_execution_indices`.

Each accepted broadcast is evaluated with row-major broadcast semantics and
encoded in a canonical little-endian representation. F32 conversion rejects
overflow and non-finite results. BF16 uses round-to-nearest-even from the
canonical F32 value and also rejects a non-finite rounded result. BOOL is one
byte and I32 is a checked signed 32-bit value. This is deliberately separate
from Python, NumPy, or framework tensor serialization.

Equal output byte strings share one content-addressed blob entry. Unique
payloads are ordered by first execution, aligned to 16 bytes, and zero padded.
The default and hard aggregate bound is 64 MiB. Every record binds the schedule
execution/site, source and output value IDs, source literal identity,
dtype/shape/layout/device, broadcast axes, byte count, blob offset, and SHA-256.
The artifact also binds the exact inventory and schedule digests.

`dump_literal_materialization(...)` emits deterministic, payload-free JSON. It
contains the data digest and all addressing receipts but not the raw bytes. The
caller packages `materialization.data` as its own constant blob and binds that
blob together with the JSON receipt.

Passing the result to
`build_memory_plan(schedule, literal_materialization=materialization)` marks
the generated values as constant allocations at their blob offsets and lists
their broadcasts in `elided_ops`. The memory-plan digest then binds the
materialization digest. The planner rejects a record that differs from its
scheduled op, and provider lowering must run against this final memory plan.

The same memory stage treats a row-major broadcast as a zero-copy alias only
when it inserts axes of size one and maps every source dimension to an equal
target dimension in increasing order. Any mapped expansion or any unmapped
axis larger than one remains runtime work.

This contract does not materialize model weights, synthesize runtime inputs, or
make unresolved operations executable. Artifact section assignment and device
upload are later executable-AIM/runtime stages.
