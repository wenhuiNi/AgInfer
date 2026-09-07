# BF16 GELU / multiply fusion

`aginfer compile --fuse-gelu-mul` replaces an exclusive BF16
`gelu(approximation="tanh") -> mul` with one AOT kernel. It combines with
`--fuse-projections` and `--constant-evaluator PATH`. The source ProgramIR,
weights and GEMM problems are unchanged, so algorithm selection does not need
this flag. Selection must still bind the CUBIN used for compilation.
The module must contain `aginfer_gelu_tanh_mul_bf16`; missing symbols fail
Prepare explicitly. Omitting the flag retains the unfused path and supports
previous modules/AIMs. There is no runtime fusion switch or fallback.

The lowering matches dataflow, not model or weight names. The intermediate
must have exactly one consumer and not be a public output. All three boundary
tensors and the intermediate must have identical contiguous CUDA BF16 types,
rank 1..4 and 1..67,108,864 elements. Broadcasting, other approximations,
shared/exported intermediates and F32 remain unfused. When both multiply
inputs are activated, only one pair claims that multiply. Existing dependency
ordering schedules the other input before the fused boundary; command-level
liveness removes the eliminated intermediate allocation.

The kernel uses the existing GELU implementation, **rounds its result to BF16**,
then multiplies in F32 and rounds to BF16 as the original multiply did. It does
not combine gate/up GEMMs, alter the activation formula, relax compiler math
flags or remove an intermediate rounding. The fixed 128-byte `AIGMU1` payload
binds SM120, element count and module identity. Bind checks access, size,
alignment and output/input overlap. Inputs may alias each other, but output
may overlap neither input. Capture records one native launch; candidate build
capabilities are not promoted to validated receipts by the compiler.

## Acceptance and scope

CPU tests cover structural selection/refusal, operand order, unique ownership,
dead intermediate memory and canonical bounded payloads. Optional
`gelu_mul_cuda_test` compares against the original two CUDA kernels at 37,
204800 and 15859712 elements, including all finite BF16 input bit patterns at
the two real FFN sizes, eager plus two poisoned Graph replays, and bad binding
rejections. These are synthetic kernel contracts, not model accuracy data.

PI0.5 real-fixture E2E-A exercised 198 fused sites (180 denoise + 18 prefix),
5959 -> 5761 commands and 7368 -> 7170 Graph nodes. Final actions, changed
image/noise and restored inputs were bit-exact against the constant-folded
baseline; original official-output gates passed and fallback was zero.
GEMM/attention algorithm payloads were checked unchanged except module identity.
Arena/workspace budgets remained unchanged.

Three alternating pairs, both using Graph, measured median 63.989786 ->
62.721064 ms (~1.98% lower). This is preliminary short-run evidence: offline
artifact verification was also running on the host. It is not an isolated
performance certification, a P95/P99 estimate or a closed-loop robotics result.
Do not combine percentages from separate runs. Heavy model/fixture/AIM/harness
files remain outside Git.
