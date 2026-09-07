# Tagged command stream v1

The tagged command stream is the bounded binary handoff from provider lowering
to a native runtime parser. It is independent of the AIM section directory: an
AIM variant may eventually contain one command-stream section, but this codec
does not imply that a provider is installed or that a model is executable.

`CommandStream.to_bytes()` emits little-endian packed fields with no native
pointers, host strings, or implicit padding. Commands execute in record order.
Each operand identifies a value in the corresponding static execution schedule
by numeric ID and declares `read`, `write`, or `read_write` access plus a checked
byte offset. Value sizes and physical base addresses remain owned by the memory
plan; a later lowering gate must cross-check an operand offset against that
plan before emitting an executable artifact.

## Fixed header

The 192-byte header uses this packed layout:

| Offset | Type | Field |
|---:|---|---|
| 0 | `u8[8]` | magic `AICMD1\0\0` |
| 8 | `u16` | schema major, currently 1 |
| 10 | `u16` | schema minor, currently 0 |
| 12 | `u32` | header size, exactly 192 |
| 16 | `u32` | exact CUDA architecture enum |
| 20 | `u32` | command count |
| 24 | `u32` | operand count |
| 28 | `u32` | header flags, currently zero |
| 32 | `u32` | reserved, zero |
| 36 | `u64` | execution-schedule value count |
| 44 | `u64` | arena bytes |
| 52 | `u64` | persistent state bytes |
| 60 | `u64` | provider workspace bytes |
| 68 | `u64` | command-table offset, exactly 192 |
| 76 | `u64` | operand-table offset |
| 84 | `u64` | payload offset |
| 92 | `u64` | total payload bytes |
| 100 | `u64` | complete section size |
| 108 | `u8[32]` | nonzero SHA-256 identity of the memory plan |
| 140 | `u8[32]` | SHA-256 of all command, operand, and payload bytes |
| 172 | `u8[20]` | reserved, zero |

The three body tables are canonical and contiguous: fixed-size command records,
then fixed-size operands, then opaque provider payloads. Counts, multiplication,
offsets, total size, and the body digest are checked before records are decoded.
The future AIM section digest must additionally cover this header as part of the
whole section.

## Command record

Each command record is 96 bytes:

| Offset | Type | Field |
|---:|---|---|
| 0 | `u32` | command tag |
| 4 | `u32` | positive numeric provider ID |
| 8 | `u32` | positive provider ABI major |
| 12 | `u32` | provider ABI minor |
| 16 | `u32` | command flags; bit 0 means capture-safe |
| 20 | `u32` | first operand index |
| 24 | `u32` | operand count |
| 28 | `u32` | payload size, at most 64 MiB per command |
| 32 | `u64` | payload offset relative to the payload table |
| 40 | `u64` | workspace offset |
| 48 | `u64` | workspace bytes |
| 56 | `u8[32]` | exact provider-capability receipt digest |
| 88 | `u8[8]` | reserved, zero |

Known v1 tags are `cuda_kernel=1`, `cublaslt_matmul=2`, `attention=3`, and
`memory_copy=4`. The payload is not self-describing JSON: only the implementation
selected by the exact `(tag, provider_id, ABI, capability digest)` tuple may
decode it. Every command has at least one write operand. Workspace slices must
fit the header's global workspace bound; zero-size slices use offset zero.

## Operand record

Each operand record is 16 bytes:

| Offset | Type | Field |
|---:|---|---|
| 0 | `u32` | numeric value ID |
| 4 | `u32` | access: `read=1`, `write=2`, `read_write=3` |
| 8 | `u64` | byte offset from that value's memory-plan base |

Command operand and payload spans must consume their complete tables in order;
overlap, gaps, aliases through duplicate table spans, and trailing bytes are not
accepted. Unknown schema versions, architectures, tags, access modes, flags,
reserved bytes, out-of-range value IDs, invalid workspace slices, truncated
records, and hash mismatches fail closed.

The Python codec deliberately does not resolve provider capabilities, assign
memory, inspect provider payloads, or execute commands. Those gates stay
separate so a successfully decoded stream cannot be mistaken for an executable
model.

The first provider-specific payload contract is the still fail-closed
[`cuBLASLt linear payload`](cublaslt-linear.md). Its syntax is independently
versioned, while the command record binds it to an exact provider ABI and
capability digest.
