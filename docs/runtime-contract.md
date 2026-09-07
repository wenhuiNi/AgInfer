# Experimental runtime contract

AgInfer's deployment boundary is the C API in `include/aginfer/c_api.h`.
`include/aginfer/runtime.hpp` is a header-only C++17 RAII wrapper over the same
functions; it is not a separate C++ ABI. Both the artifact format and runtime
ABI remain experimental; numerical model validation alone does not freeze a
public release ABI.

## Versions and fail-closed rules

- `AI_RUNTIME_ABI_VERSION` identifies the callable C ABI. An AIM with another
  runtime ABI is rejected at load time.
- Every extensible C input/output struct starts with `struct_size` and
  `struct_version`. Call its `*_init` function before use. A short struct, an
  unknown version, non-zero unknown flags, or an invalid enum is rejected.
- Experimental AIM v2, compatibility-table v1, and execution-plan v1/v2 have
  independent fixed headers. Unknown major versions, newer minor versions,
  non-zero reserved bytes, oversized counts, and out-of-bounds tables fail
  closed. Prototype AIM v1 files are intentionally incompatible.
- Target selection is exact. The host platform and CUDA architecture must have
  an explicit variant; there is no cross-architecture fallback or PTX JIT.
- CUDA Driver/Runtime and provider ABI requirements live in the checksummed
  binary compatibility section. The human-readable manifest is not consulted
  for runtime compatibility decisions.

The AIM header's file checksum (also exposed as `AimInfo.file_sha256`) hashes
the file with its own 32-byte checksum field at offset 152 zeroed. It is not
the raw-file SHA-256 produced by `sha256sum`; validation reports should label
these two identities separately.

### Load-time checksum policy

Full AIM container checksum scans are enabled automatically for CMake `Debug`
builds. Other configurations (including `Release` and `RelWithDebInfo`) skip
these scans by default. To build a verification runtime in any configuration,
set `-DAGINFER_VERIFY_AIM_CHECKSUMS=ON`; setting it `OFF` does not disable the
automatic checks in `Debug`. This is a compile-time policy, not an enqueue or
session option.

Fast loading still checks section bounds, header versions/reserved fields,
compatibility records, variant bounds and exact CUBIN architecture. It does not
detect arbitrary weight/content corruption that preserves those structures.
Use the offline `aginfer verify` command on artifacts before deployment; its
full checksum verification is independent of the C++ build configuration.
Neither path authenticates an untrusted artifact or permits modifying a mapped
AIM while it is in use.

This policy applies to container checksums in `ai_model_load`. Small binary
plan/command integrity checks at session creation and module identity checks at
prepare remain enabled; fast loading does not remove those execution contracts.

## Ownership and lifetime

`ai_runtime_create`, `ai_model_load`, and `ai_session_create` return opaque,
caller-owned handles. Destroy them with the matching function. Destroying a
null handle is allowed. A runtime and model must outlive every session created
from them, and a session must be externally synchronized rather than used
concurrently.

Strings returned through `ai_port_info.diagnostic_name` remain valid while the
model and session remain alive. Tensor storage is always caller-owned: binding
copies the pointer and validates the static contract but does not retain a C
object or copy the buffer. The pointer must remain valid through every enqueue
that uses the binding.

## Execution phases

1. `ai_model_load` mmaps and validates the container structure, binary
   compatibility table, variants, and exact CUBIN architecture. Debug and
   verification builds additionally check all container checksums. It does not
   create a CUDA context or upload weights.
2. `ai_session_create` selects one exact target/profile, validates provider
   compatibility, and parses the bounded binary execution plan.
3. Bind inputs and outputs by compile-time numeric port ID. Names are for
   inspection and diagnostics only. Execution-plan v2 requires all ports bound
   before prepare; their addresses then stay fixed for the session lifetime.
   Updating buffer contents does not require rebinding.
4. `ai_session_prepare` loads the CUBIN, uploads weights, allocates fixed
   arena/state/workspace, resolves functions, and reconstructs fixed library
   descriptors and algorithms. V2 library commands are recorded on a temporary
   nonblocking capture stream to trigger deferred library initialization. The
   resulting graph is destroyed without instantiation or launch: no caller
   input is read, no caller output is written, and execution counters remain
   zero. No tactic search is performed. A failed prepare leaves the session
   unusable; recreate it rather than retrying a partial initialization.
5. `ai_session_enqueue` requires a prepared session and all selected-profile
   ports to be bound. It launches on the caller-provided CUDA stream without an
   implicit device synchronization.

Execution-plan v1 launches fixed CUDA kernels directly. V2 dispatches a bounded
tagged command stream through prepared native providers; enqueue itself does
not parse plans, load modules, allocate device memory, choose algorithms, or
synchronize. Vendor libraries can still perform cached kernel-handle queries;
these are not a promise of lookup-free third-party internals. Prepare-time
capture preflight is not a deployment CUDA Graph replay mode.
