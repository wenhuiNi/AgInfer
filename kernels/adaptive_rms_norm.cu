#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kRows = 50;
constexpr int kWidth = 1024;
constexpr float kEpsilon = 1.0e-6F;

__device__ __forceinline__ float LoadBf16(const __nv_bfloat16* input,
                                          std::uint64_t index) {
  return __bfloat162float(input[index]);
}

}  // namespace

// Exact adaptive RMSNorm form used by fixed-width transformer blocks:
//   normalized = bf16(rms(f32(hidden)) * (1 + scale) + shift)
//   gate       = bf16(broadcast(gate_vector))
// modulation is laid out as contiguous [scale, shift, gate] vectors.
template<bool WriteGate>
__device__ __forceinline__ void AdaptiveRmsNorm(
    const __nv_bfloat16* hidden, const float* modulation,
    __nv_bfloat16* normalized, __nv_bfloat16* gate) {
  __shared__ float reduction[256];
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  if (row >= kRows) return;

  const std::uint64_t row_offset = static_cast<std::uint64_t>(row) * kWidth;
  float sum_of_squares = 0.0F;
  for (int column = lane; column < kWidth; column += blockDim.x) {
    const float value = LoadBf16(hidden, row_offset + column);
    sum_of_squares += value * value;
  }
  reduction[lane] = sum_of_squares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(kWidth) + kEpsilon);
  for (int column = lane; column < kWidth; column += blockDim.x) {
    const float value = LoadBf16(hidden, row_offset + column);
    const float scale = modulation[column];
    const float shift = modulation[kWidth + column];
    normalized[row_offset + column] =
        __float2bfloat16_rn(value * inverse_rms * (1.0F + scale) + shift);
    if constexpr (WriteGate) gate[row_offset + column] =
        __float2bfloat16_rn(modulation[2 * kWidth + column]);
  }
}

extern "C" __global__ void aginfer_adaptive_rms_norm_bf16_f32_1024(
    const __nv_bfloat16* hidden, const float* modulation,
    __nv_bfloat16* normalized, __nv_bfloat16* gate) {
  AdaptiveRmsNorm<true>(hidden,modulation,normalized,gate);
}
extern "C" __global__ void aginfer_adaptive_rms_norm_no_gate_bf16_f32_1024(
    const __nv_bfloat16* hidden, const float* modulation, __nv_bfloat16* normalized) {
  AdaptiveRmsNorm<false>(hidden,modulation,normalized,nullptr);
}
extern "C" __global__ void aginfer_modulated_residual_bf16_f32_1024(
    const __nv_bfloat16* activation, const float* modulation,
    const __nv_bfloat16* residual, __nv_bfloat16* output) {
  const auto i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<51200) {
    const auto gate=__float2bfloat16_rn(modulation[2048+(i%1024)]);
    const auto product=__float2bfloat16_rn(__fmul_rn(__bfloat162float(activation[i]),__bfloat162float(gate)));
    output[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(product),__bfloat162float(residual[i])));
  }
}
