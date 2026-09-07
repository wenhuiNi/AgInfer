# Experimental source and constant contract

The source layer is an offline compiler boundary. It inventories checkpoint
files without executing remote code or deserializing pickle, and it keeps
model, preprocessor, postprocessor, tokenizer, and external assets in explicit
namespaces. Nothing in this layer is a deployment Runtime dependency.

`SourceManifest` records safe relative paths, exact file sizes, tensor metadata,
logical shard sets, and byte ranges. A logical tensor address is the pair
`(namespace, name)`; a shard filename is storage metadata and is not part of a
recipe slot. A deserialized manifest fails closed on duplicate logical tensor
addresses, unsafe paths, unknown dtypes or roles, asset/tensor namespace
disagreement, and ranges outside the declared file.

`ConstantStore` resolves the same logical address for either one safetensors
file or an indexed shard set. Construction verifies required assets and exact
sizes without reading tensor payloads. Payload access is copy-bounded or a
scoped read-only mmap view, and it rechecks safetensors metadata against the
manifest before exposing bytes.

`ConstantCoverage` is the recipe-side ledger. Each source tensor must be marked
exactly once as consumed by a semantic slot or explicitly ignored with a
non-empty reason. Finalization fails while any tensor is unresolved. It does
not infer matches from model IDs or silently discard auxiliary state.

`open_source_package` resolves a local or explicitly enabled remote source into
one `SourceManifest`, `ConstantStore`, and bounded `AssetBundle`. The asset
bundle reads only manifest-declared files, enforces an explicit byte limit, and
can parse UTF-8 JSON without executing source code. `SourceFrontend` is a
structural importer protocol. `FrontendOutput.finalize` accepts only verified
ProgramIR and complete coverage belonging to the exact source store.

`inspect_source_asset_contract` turns feature-bearing model metadata and
processor JSON into immutable IO, ordered-step, state-file, and external-asset
contracts. Processor discovery uses namespace or the joint `name`/array-`steps`
structure; a training counter named `steps` is not sufficient. State files must
resolve to a declared asset in the same pre/post namespace. Referenced external
tokenizers are reported as requirements and are never downloaded implicitly.

`Pi05SourceFrontend` is the first concrete recipe. For the locked PI0.5
tensor-only profile it validates the feature/processor asset contract and exact
tensor inventory, builds the full vision/prefix/cache/ten-step-denoise
ProgramIR directly from canonical config, and finalizes against the same
`ConstantStore`. It neither loads tensor payloads nor imports the originating
framework. The recipe records an external tokenizer requirement but does not
download it. This is an importer contract, not a claim that provider lowering,
AIM generation, or Runtime execution is complete.

Payload digests and normalized constant identities belong to the compiler
build-record boundary, where their cost and lifetime are explicit.
