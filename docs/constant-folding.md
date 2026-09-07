# Offline native constant folding

After selecting fixed algorithms, `aginfer compile --constant-evaluator PATH`
can remove eligible constant-only native subgraphs. `PATH` is the offline
`aginfer_constant_eval` executable built with `AGINFER_BUILD_COMPILER_TOOLS=ON`
and CUDA enabled. Selection options (`--fuse-projections`, `--resident-kv`)
still need to match compilation; folding runs **after** selection and does not
require another tactic search. Omit `--constant-evaluator` to retain the baseline.

## Numerical and state contract

The initial envelope covers delivered F32 row-one cuBLASLt linears, time embedding,
and SiLU. Discovery follows immutable constant roots, native operand dependencies,
single-writer storage and exact payload identities, not model/function names.
External inputs, mutable state, nonzero operand offsets, read/write operands,
unknown forms and public output writes are not folded. A requested compilation
with no eligible boundary fails explicitly.

The compiler constructs a temporary zero-input evaluation AIM using the **same
native payloads, packed source values and CUBIN** as the unoptimized commands.
The evaluator executes its graph twice, poisons outputs between runs, requires
finite F32 output, bit-exact repeat and the expected native submission ledger.
It does not use CPU/PyTorch approximations, calibration or quantization. V2 plans
now permit zero input ports; at least one output and complete external-storage
binding are still required. The native parser and Python parser agree.

Only values consumed outside the folded region are retained, not broadcasted
copies of them. These become read-only packed constants. Existing command-level
topological ordering/liveness is rerun; retained commands and their memory plan
use the same ordering. No runtime computation, new state cache, tactic search or
weight repacking is added. Source inventory/ProgramIR identity describes the
pre-fold graph; the folding report and final command/memory/weight identities
describe the post-selection executable transformation.

The report binds the evaluator binary, evaluation plan/weights, original command
stream, removed count, boundary value IDs/sizes/hashes and concatenated output.
The enclosing build binds checkpoint, algorithm and kernel identities. Offline
`verify --require-build-record` checks the report against packed executable
values and their bytes. This is provenance and repeat evidence, **not** automatic
model-level numerical/performance certification. Source changes require fresh
compilation; arbitrary caller-supplied cached constants are not a CLI input.
Limits: 4096 evaluated commands, 16 MiB total intermediate output budget,
2 GiB evaluation AIM and a 120-second helper timeout. Temporary evaluation files
are removed with compilation scratch; only their identities enter the final AIM.

## PI0.5 result (single fixture, SM120)

The locked ten-step profile removes 390 small F32 linears, ten sinusoidal
embeddings and twenty SiLUs: 6379 -> 5959 commands. It retains 370 modulation
vectors totaling 4,546,560 bytes. Dead time-only source weights are no longer
packed; AIM size decreased from 6,951,263,168 to 6,481,280,960 bytes.

With both arms using CUDA Graph, three alternating pairs measured median
64.846067 -> 61.598162 ms (~5.01% lower). Final actions were bit-exact against
the unfolded native path, including changed image/noise and restoration, and
passed the original official-output gates with zero fallback. This is a small
tensor-only E2E-A experiment, not a distribution-wide or closed-loop claim.
Heavy checkpoint/fixture/AIM/profile artifacts are deliberately outside Git.
