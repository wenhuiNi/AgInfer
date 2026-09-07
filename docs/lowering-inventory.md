# Lowering inventory

`aginfer.lowering.build_lowering_inventory(program)` is the payload-free boundary
between verified ProgramIR and provider selection. It verifies the program, walks
the fixed call schedule, and records each static op site together with its input
and output tensor signatures, canonical attributes, function invocation count,
and expanded runtime order.

The v1 inventory uses two separate axes:

- `lowering_kind` is the candidate implementation family:
  `builtin_meta`, `memory`, `gemm`, `attention`, `aot_cuda`, or `blocked`.
- `status` is `meta`, `required`, or `blocked`. `required` means that compilation
  must resolve a matching provider capability later; it does **not** claim that a
  provider is installed or that the op is executable.

`constant_ref` records preserve one `namespace:name` identity per static site.
Repeated function calls increase the site's `executions` count without cloning
the constant identity or implying another weight payload. `call`, `constant`, and
`constant_ref` are compile-time metadata and do not appear in
`execution_order`; memory and provider-requiring ops do.

The current E2 inventory classifies non-overlapping, unpadded patch-projection
`conv2d` as a GEMM requirement. Other convolution envelopes and non-CUDA tensor
execution are explicitly `blocked` with a reason. No fallback provider is
inferred. Expansion is bounded by `max_expanded_ops` so an excessive nested
`call.repeat` fails before producing an unbounded report.

`dump_lowering_inventory(inventory)` emits deterministic canonical JSON. The
report embeds the SHA-256 of the canonical ProgramIR dump, per-op static and
expanded counts, and the complete execution-order receipt. It contains logical
constant addresses and tensor metadata only; it never reads or embeds tensor
payloads.

## Provider capability resolution

`ProviderCapability` is an exact target record, not a fuzzy advertisement. It
declares the numeric provider ID, ABI major/minor, provider version,
implementation ID and nonzero implementation/payload SHA-256, exact CUDA
architecture, lowering kind, opcode, complete
input/output tensor signatures, semantic attributes, capture support, and
workspace bound. `ProviderCapability.exact_for(...)` can derive the semantic
part from one `required` inventory site without accessing constant payloads.

`resolve_provider_capabilities(...)` matches only those declared fields. Model,
function, and SSA names are not match inputs. Registration order is not a
priority mechanism: zero matches produces an `unresolved` issue and multiple
matches produces an `ambiguous` issue. With the default `strict=True`, either
case fails compilation. `strict=False` returns the same deterministic issue
receipt for inspection, but does not make the inventory executable. An
inventory-level blocker cannot be overridden by registering a capability.

Receipts are per static op site and carry the site's expanded execution count,
so repeated calls reuse one capability identity while retaining the expected hit
count. The implementation digest binds the capability identity to the exact
provider payload or target code instead of treating two tactics with equal
workspace as interchangeable. Requiring that field changed the experimental
capability and resolution report schemas to v2. `dump_provider_capabilities(...)` and
`dump_provider_resolution(...)`
emit canonical JSON with content digests for artifact and ledger work in later
stages. A capability receipt is still not a tagged command or executable
provider; those are separate lowering and validation gates.

## Call-inline value schedule

`build_execution_schedule(program)` consumes the verified inventory and inlines
the entry function's fixed `call`/`repeat` schedule. Every runtime op receives a
monotonic execution index and every operand is rewritten to a global numeric
value ID. Repeated calls thread one invocation's outputs into the next
invocation's inputs exactly as ProgramIR specifies; the `call` itself is not a
runtime command.

Values have explicit storage classes: `entry_input`, `constant`, `state`,
`state_alias`, and `temporary`. Literals are shared by static site across
repeated invocations; `constant_ref` values are additionally deduplicated across
static sites by their exact `namespace:name` logical identity, with inconsistent
types rejected. Ordinary SSA outputs get a fresh identity for every invocation. A state has one persistent identity; each
`state_read` result is a declared alias of that identity and `state_write`
remains an ordered schedule op. This preserves state semantics for later arena
planning rather than hiding them in compiler-side mutation.

The schedule records invocation parent/call-site/repeat receipts, entry IO,
states, values, ops, the ProgramIR digest, and the exact inventory digest.
Construction checks that its runtime site order equals the independently built
inventory order and that every operand/producer/root is defined exactly once.
`dump_execution_schedule(...)` emits canonical JSON. It assigns identities but
does not yet choose byte offsets, aliases for ordinary transforms, provider
commands, or an AIM encoding.

## Static memory plan

`build_memory_plan(schedule)` requires every value shape to be specialized and
computes its checked uint64 byte size. Entry inputs and constants remain in
external/constant storage, state values receive a separate persistent region,
and ordinary temporaries receive liveness intervals in a reusable arena. The
default alignment is 256 bytes and must be a bounded power of two.

The allocator expires a value only when its last-use op is strictly earlier
than the next producer. It therefore never reuses an input's storage for an
output of the same op unless a future op contract explicitly permits in-place
execution. Entry outputs remain live through the end of the schedule. Allocation
is deterministic first-fit over merged free spans; the plan reports arena size,
peak live bytes, naive per-value bytes, and reuse savings.

Contiguous row-major `reshape` is an explicit alias and is listed in
`elided_ops`; alias uses extend the root value's lifetime. A
`broadcast_in_dim` is also an alias when it only inserts size-one axes and
preserves every mapped dimension and the row-major physical order. Any real
expansion remains an operation. `state_read` is an elided alias of dedicated
state storage, while `state_write` remains a real ordered operation. Every scheduled value gets exactly one `external`,
`constant`, `state`, `arena`, or `alias` disposition. The planner verifies arena
bounds and checks all simultaneously live intervals for overlap before
`dump_memory_plan(...)` emits canonical JSON. Output-direct placement, general
slice aliases, provider workspace, and command encoding are intentionally later
contracts.

Residual broadcasts from verified static literals can be evaluated offline and
attached to this plan as a deduplicated constant blob. The exact encoding,
exclusion, identity, and fail-closed rules are documented in
[`literal-materialization.md`](literal-materialization.md).

The next binary boundary is documented in
[`command-stream.md`](command-stream.md). Decoding that format does not change
an unresolved capability into an executable provider command.
