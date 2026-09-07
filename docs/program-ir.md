# Experimental ProgramIR v1 contract

ProgramIR is AgInfer's framework-independent semantic layer. The current v1
implementation is deliberately small: it exists to make type, SSA, shape, and
state rules executable before source frontends or provider lowering are added.

## Ownership and versioning

`Program`, `Function`, `Region`, `Op`, `Value`, `TensorType`, `State`, and
`ShapeDomain` are immutable Python records owned by the offline compiler. They
must not contain framework objects, CUDA symbols, model IDs, mutable runtime
handles, or checkpoint-specific module names.

ProgramIR has an independent `schema_major/schema_minor`. The verifier accepts
major version 1 and known minor versions only. This Python representation is
not the future artifact encoding; AIM serialization will get a separate,
bounded binary contract after provider and compiler requirements are known.

## Verified semantics

- Values are typed SSA definitions. Inputs must be defined before use, every
  value is defined once, and function outputs must name live values.
- A `TensorType` carries dtype, static or symbolic shape, logical layout, and
  device. Every symbolic dimension must have one positive min/preferred/max
  range in the program `ShapeDomain`.
- State is declared separately from function SSA values. `state_read` produces
  the declared type; `state_write` requires that type and `read_write` access.
- The initial generic ops are `constant`, `constant_ref`, `cast`, `add`, `mul`, `logical_and`,
  `cumulative_sum`,
  `matmul`, `linear`, `reshape`, `broadcast_in_dim`, `transpose`, `concat`, static `slice`, `reduce_sum`,
  `reduce_max`, `gather`, `rms_norm`, `layer_norm`, `softmax`, `relu`, `silu`,
  `gelu`, `sinusoidal_embedding`, `rope`, `rope_default`, `scaled_dot_product_attention`, and `conv2d`, plus explicit
  state read/write. Gated MLPs and patch projections are represented by
  compositions of these ops rather than model-specific compound operations.
  Tensors may be rank zero;
  each opcode has an exact input/output/attribute/type contract, and unknown
  ops or attributes fail closed.
- `constant_ref` addresses large source tensors by `(namespace, name)` rather
  than embedding payloads in IR. `FrontendOutput.finalize` checks its dtype,
  shape, and consumed coverage against the exact `ConstantStore`.
- `broadcast_in_dim` carries an explicit increasing input-to-output axis map.
  GELU requires an explicit `none` or `tanh` approximation, and time
  sinusoidal embeddings carry their dimension and period range as attributes.
- `logical_and` requires exact BOOL types. `cumulative_sum` has an explicit
  static axis and preserves a non-BOOL numeric tensor type; mask-to-position
  conversion therefore requires an explicit cast.
- `call` names another ProgramIR function. The call graph must be acyclic and
  signatures match exactly. A positive fixed `repeat` can feed each call's
  outputs into the next call when the callee input/output signatures are the
  same; data-dependent or host-driven loops are not represented.
- `rope` takes explicit position IDs and cosine/sine tables and declares either
  interleaved or split-half pairing. `scaled_dot_product_attention` takes
  explicit q/k/v, an exact BOOL mask, and a `kv_group_size` for MHA or GQA; it
  does not infer causal/cache policy or mask broadcasting. Its required
  `mask_fill="dtype_min"` contract replaces masked logits with the finite minimum
  of the query dtype. Consequently an all-false query row has equal finite logits
  and produces the uniform mean of all value rows; providers must not substitute
  negative infinity, zero output, or rejection for that case.
- `rope_default` is the position-driven default rotary form. It takes an exact
  I32 position tensor and declares pairing, positive finite `theta`, and a
  floating `frequency_dtype`. The generated `theta^(-2i/D)` inverse-frequency
  vector is rounded to that dtype before position multiplication. This dtype is
  semantic: some BF16 loading paths convert the registered inverse-frequency
  buffer before forward, and silently recomputing it as FP32 changes deep-model
  results. Non-default/scaled variants continue to use explicit-table `rope`
  until their generic semantics are separately specified.
- `rms_norm` permits an FP32 weight with FP16/BF16 activations, preserving the
  activation output dtype. This represents the common FP32 statistic/parameter
  path without silently downcasting the parameter. Any `1 + weight` convention
  remains explicit dataflow outside the normalization op.
- `dump_program` verifies first and emits a deterministic text form for review
  and pass diffs.

The pure-Python reference executor uses immutable flat row-major tensors. It is
an oracle for small CPU fixtures, not a performance implementation. It checks
concrete symbolic bounds and shared-symbol consistency, rejects non-finite
floating data and implicit integer truncation, and returns state updates
explicitly. FP16/BF16 values currently use Python arithmetic and therefore do
not yet model format-specific rounding.

The current `conv2d` contract is NCHW with groups and dilation fixed to one.
The contract does not yet include a complete KV-cache recipe, calibration,
serialization, or lowering. No supported-model claim follows from these small
generic graphs.
