# Experimental materialized attention

This explicit correctness-diagnostic executable form is **not yet an approved
model backend**. It has no automatic capability registration or fallback from
FlashInfer. Command provider ID 5, ABI 1.0, and the attention tag identify it
separately in the submission ledger.

The two BF16 executable forms preserve BF16 rounding of QK scores, scaling,
probabilities and the final PV result, with F32 accumulation:

- Payload v1: a single-launch scalar implementation, not a claim of
  bit-identical vendor GEMM reduction order.
- Payload v2: fixed cuBLASLt QK, an explicit rounded softmax, and fixed
  cuBLASLt PV. This is three launches, not a single-kernel fusion. The PV
  output uses interleaved batch strides to write BSHD directly.

The two SM120 envelopes use batch 1, Q heads 8, KV heads 1, head dimension 256,
BF16 Q/K/V/output, scale 0.0625, and BOOL masks:

- Q length 968, K length 968: shared pad mask `[968]`; validity is the outer
  product of the query and key mask.
- Q length 50, K length 1018: dense head-broadcast mask `[50,1018]`.

Input Q/K/V are BHSD; output is BSHD. Fully masked rows have uniform
probabilities, matching finite mask-fill semantics. No implicit cache state,
allocation, synchronization, or conversion occurs in Execute.

The fixed 128-byte payload stores schema, target, variant, dimensions, and
module size/SHA-256. Dtypes, layouts, scale and launch geometry are fixed by
the variant. Unknown fields, trailing bytes, reserved bits, module mismatch,
short/misaligned buffers, and overlapping output bindings are rejected.

The 192-byte v2 payload additionally fixes the exact cuBLASLt version, all
nine algorithm fields for each matmul, and shared scratch of `8*Q*K*2`
bytes. The score buffer is overwritten by BF16 probabilities. Prepare
reconstructs descriptors and algorithms and checks them with AlgoCheck;
neither Prepare nor Execute runs heuristic selection. These algorithms
require zero additional cuBLASLt workspace. Wrong library versions,
unsupported algorithms and scratch overlap fail explicitly. The v2 form
does not silently fall back to v1 or FlashInfer.

Two additional explicit v2 variants accept F32 BSHD Q/K/V/output with
16 heads, head dimension 72, and Q/K length 256. It uses the same three-call
library form: variant 3 fixes TF32 matmul compute and variant 4 fixes FP32.
Both preserve F32 score scaling/softmax,
scale `1/sqrt(72)`, and a shared scalar BOOL mask. Its scratch is
`16*256*256*4` bytes. Fully masked rows retain uniform finite-min semantics.
There is no implicit conversion between these forms or the BF16 variants.
TF32 permission for independent linear layers does not imply that a source
SDPA implementation uses TF32 attention; validate those boundaries separately.

Selection for deployment still requires real-boundary and complete-model
gates; a successful diagnostic run alone does not qualify this backend.
