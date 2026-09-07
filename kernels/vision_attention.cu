#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kVisionSequence = 256;
constexpr int kVisionHeads = 16;
constexpr int kVisionHeadDim = 72;
constexpr float kVisionScale = 0.11785113019775793F;

__device__ __forceinline__ std::uint64_t BshdOffset(int token, int head,
                                                    int dimension) {
  return (static_cast<std::uint64_t>(token) * kVisionHeads + head) *
             kVisionHeadDim +
         dimension;
}

}  // namespace

// Exact PI0.5 vision attention executable form. Q/K/V arrive in BSHD so the
// three input transposes are fused, and output is written in BSHD so the
// attention output transpose is fused as well. The one-byte mask is the
// pre-broadcast [1] BOOL value. A false mask intentionally gives uniform
// softmax weights, matching finite dtype_min mask-fill semantics.
extern "C" __global__ void aginfer_vision_attention_f32_bshd(
    const float* query_bshd, const float* key_bshd,
    const float* value_bshd, const std::uint8_t* mask_scalar,
    float* output_bshd) {
  __shared__ float query[kVisionHeadDim];
  __shared__ float weights[kVisionSequence];
  __shared__ float reduction[kVisionSequence];

  const int query_token = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int lane = static_cast<int>(threadIdx.x);

  if (lane < kVisionHeadDim) {
    query[lane] = query_bshd[BshdOffset(query_token, head, lane)];
  }
  __syncthreads();

  float score = 0.0F;
  if (mask_scalar[0] != 0) {
#pragma unroll
    for (int dimension = 0; dimension < kVisionHeadDim; ++dimension) {
      score += query[dimension] *
               key_bshd[BshdOffset(lane, head, dimension)];
    }
    score *= kVisionScale;
  }
  weights[lane] = score;
  reduction[lane] = score;
  __syncthreads();

  for (int stride = kVisionSequence / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  // Every thread must consume the maximum before lane 0 reuses reduction[0]
  // for the softmax sum. Without this barrier, independent warp scheduling
  // makes a small set of query/head blocks nondeterministic.
  __syncthreads();
  const float exponential = expf(score - maximum);
  weights[lane] = exponential;
  reduction[lane] = exponential;
  __syncthreads();

  for (int stride = kVisionSequence / 2; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];

  if (lane < kVisionHeadDim) {
    float result = 0.0F;
    for (int key_token = 0; key_token < kVisionSequence; ++key_token) {
      result += weights[key_token] * inverse_sum *
                value_bshd[BshdOffset(key_token, head, lane)];
    }
    output_bshd[BshdOffset(query_token, head, lane)] = result;
  }
}
