# AIM schema 1.0

AIM v1 is a little-endian, mmap-friendly static container. Every integer is
unsigned and little-endian. All payload sections start at a 256-byte boundary.
There is no PTX section and loaders must reject PTX text in kernel bundles.

## File layout

| Region | Size | Purpose |
|---|---:|---|
| Fixed header | 256 bytes | Schema, ABI, target platform, section locations and hashes |
| Manifest | variable | Canonical UTF-8 JSON model/target/profile metadata |
| Graph | variable | Canonical UTF-8 JSON explicit graph and iterative recipe |
| Tensor table | variable | Canonical UTF-8 JSON dtype/shape/stride/layout/blob offset/SHA-256 records |
| Variant payloads | variable | Per-arch CUBIN, packed weights, and static plan; identical blobs may share an offset |
| Variant directory | 192 bytes each | Exact architecture and payload locations/hashes |

The fixed header is described by the Python `struct` string:

```text
<8sHHIIIII QQQQQQQQQ 32s32s32s32s24s
```

Its magic is `AIMAOT1\0`. Schema major changes are incompatible. Runtime ABI
must match exactly. Platform IDs are `1 = linux-x86_64-gnu` and
`2 = linux-aarch64-sbsa`; a package never contains both.

The 32-byte file digest starts at byte 136. It is SHA-256 over the complete
file with those 32 bytes treated as zero. The directory, manifest, graph, and
each variant payload also have independent SHA-256 values. Readers validate
bounds before reading or hashing any region.

## Variant directory

Each 192-byte entry is:

```text
<II QQQQQQ 32s32s32s40s
```

It records the numeric compute capability (`89`, `110`, or `120`), flags
(currently zero), offset/size pairs for CUBIN, weights and plan, their hashes,
and zeroed reserved bytes. Selection is exact: an `sm120` device cannot load
an `sm110` or `sm89` variant.

## Compatibility and trust boundary

Both writer and loader enforce the supported platform matrix. Runtime loading
also checks host platform, file and section integrity, schema and Runtime ABI.
Session creation checks exact GPU architecture plus the CUDA Driver, CUDA
Runtime, cuBLASLt and cuDNN range/ABI stored in the manifest. No compatibility
fallback, PTX JIT, autotuning, or runtime weight conversion is permitted.

## Static execution plan

Each variant stores an `AIMPLN1\0` binary plan compiled from its `plan.json`.
The plan fixes profile-specific tensor contracts, Kernel symbols, launch
dimensions, argument bindings, arena size, and workspace size. Runtime does
not parse JSON or infer a launch configuration on the deployment machine.

The plan table layout and CUDA calling convention are specified in
[Kernel ABI 1.0](kernel-abi-v1.md).
