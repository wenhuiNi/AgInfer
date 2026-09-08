# Offline horizontal projection fusion

`select-algorithms --fuse-projections` and `compile --fuse-projections` enable an
opt-in IR transformation. Use the flag for both commands: selection receipts
bind the transformed program digest, so mixing modes with a selection report
is rejected. Existing compilation remains unchanged without the flag.

The initial envelope is three linear operations sharing one SSA input:

- CUDA row-major BF16 input `[1,M,K]`, `1 <= M <= 128`;
- source constant weights `[N0,K]`, `[N1,K]`, `[N2,K]`;
- biases statically proven to be positive zero;
- each output width divisible by eight, total width at most 65536.

The compiler concatenates weights on the output-channel dimension while
streaming the checkpoint into packed weights. A single existing cuBLASLt
projection produces `[1,M,N0+N1+N2]`. Three contiguous partitions lower to one
vectorized BF16 copy kernel, restoring the original output IDs and contiguous
layouts. Runtime does no concatenation, weight repacking, algorithm search, or
fallback. Precision is unchanged, but a different GEMM shape can change floating
point accumulation, so each new artifact still requires numerical validation.

Matching uses dataflow, shapes, and constant semantics, not model or module
names. Unsupported patterns retain the existing independent projections.
Derived constant descriptors and the source program digest are recorded in the
candidate build manifest. Selection defaults to heuristic-first-reconstructable;
projection fusion alone does not enable timing. The separate offline
`select-algorithms --benchmark-small-gemm` policy can benchmark eligible shapes.

The split payload is 128-byte little-endian `AIPSP1`, ABI 1.0, SM120, with rows,
three widths, exact CUBIN size/SHA-256, and zero reserved bytes. Prepare rejects
short/misaligned/overlapping tensors and module mismatches. Execute launches one
copy kernel on the caller stream. Candidate capture flags remain uncertified;
the synthetic CUDA test's successful capture is not a model capture receipt.

Design references (no upstream implementation copied):

- [vLLM QKVParallelLinear](https://github.com/vllm-project/vllm/blob/6fbb00b18874e27ba7d7adc0a3b8e93fee763ab1/vllm/model_executor/layers/linear.py): concatenated projection weights.
- [SGLang QKVParallelLinear](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py): the same shared-input projection form.
- [vLLM-Omni QwenImageCrossAttention](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py): combined projection followed by Q/K/V splitting.

These are design references, not runtime dependencies. The initial PI0.5 use
case covers denoise projections only; vision and prefix are unchanged. Existing
RoPE already fuses the BSHD-to-BHSD transpose. This change does not implement
fused attention or remove the separate split memory traffic.

Lightweight tests are in `tests/test_projection_fusion.py` and
`tests/projection_split_test.cc`. Enable `AGINFER_BUILD_CUDA_TESTS` for the
optional real-GPU split test. Checkpoints, real-model fixtures, AIMs and model
benchmark outputs belong outside the repository. Performance acceptance should
use identical inputs, the same runtime, a clean command ledger, numerical
gates, and short alternating baseline/candidate timing before larger studies.
