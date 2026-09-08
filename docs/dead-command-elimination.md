# Native dead-command elimination

Native lowering removes pure commands that cannot reach entry outputs or state
writes, after provider fusion and before final liveness allocation. This is an
offline compiler transformation; runtime, input profiles and arithmetic stay
unchanged. Existing AIM files remain executable without reinterpretation.

Dependencies follow storage aliases, not reused physical arena offsets. Every
state write is retained, even if subsequently overwritten. Unknown operations
and providers, read/write operands, partial accesses and multiply-written roots
are conservatively retained with their producers. Exporting an intermediate
feature keeps its producers alive, including for a standalone prefix entry.

Removed source coverage becomes explicitly elided; final command assembly still
requires complete source coverage. Final liveness and weight packing omit unused
buffers and unreferenced weights. No layer index or model ID drives this pass.

Public CPU tests cover transitive pruning, exported/aliased features, state
writes, unknown effects and partial writes. Model performance requires a separate
same-input paired E2E measurement, not command counts alone.
