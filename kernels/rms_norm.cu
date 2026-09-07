#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr float kRmsNormEpsilon = 1.0e-6F;

template <typename Input>
__device__ __forceinline__ float LoadRms(const Input* input,
                                         std::uint64_t index);

template <>
__device__ __forceinline__ float LoadRms<float>(const float* input,
                                                std::uint64_t index) {
  return input[index];
}

template <>
__device__ __forceinline__ float LoadRms<__nv_bfloat16>(
    const __nv_bfloat16* input, std::uint64_t index) {
  return __bfloat162float(input[index]);
}

template <typename Output>
__device__ __forceinline__ void StoreRms(Output* output, std::uint64_t index,
                                         float value);

template <>
__device__ __forceinline__ void StoreRms<float>(float* output,
                                                std::uint64_t index,
                                                float value) {
  output[index] = value;
}

template <>
__device__ __forceinline__ void StoreRms<__nv_bfloat16>(
    __nv_bfloat16* output, std::uint64_t index, float value) {
  output[index] = __float2bfloat16_rn(value);
}

template <typename Input, typename Output, int Width>
__device__ __forceinline__ void RmsNormRow(const Input* input,
                                           const float* weight,
                                           Output* output) {
  __shared__ float reduction[256];
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint64_t row_offset = static_cast<std::uint64_t>(row) * Width;
  float sum_of_squares = 0.0F;
  for (int column = lane; column < Width; column += blockDim.x) {
    const float value = LoadRms(input, row_offset + column);
    sum_of_squares += value * value;
  }
  reduction[lane] = sum_of_squares;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(Width) + kRmsNormEpsilon);
  for (int column = lane; column < Width; column += blockDim.x) {
    StoreRms(output, row_offset + column,
             LoadRms(input, row_offset + column) * inverse_rms *
                 weight[column]);
  }
}

}  // namespace

extern "C" __global__ void aginfer_rms_norm_f32_1024(
    const float* input, const float* weight, float* output) {
  RmsNormRow<float, float, 1024>(input, weight, output);
}

extern "C" __global__ void aginfer_rms_norm_bf16_2048(
    const __nv_bfloat16* input, const float* weight, __nv_bfloat16* output) {
  // Match the contiguous F32 mean's four independent accumulators and
  // ascending warp reduction before the final BF16 store. A block-wide
  // tree changes rounding at BF16 halfway points in deep residual stacks.
  __shared__ float inverse_rms;
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * 2048;
  if (lane < 32) {
    float sums[4] = {};
    for (int column = lane * 4; column < 2048; column += 128) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float x = __bfloat162float(input[base + column + j]);
        sums[j] = __fadd_rn(sums[j], __fmul_rn(x, x));
      }
    }
    float sum = ((sums[0] + sums[1]) + sums[2]) + sums[3];
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) inverse_rms = rsqrtf(sum / 2048.0F + kRmsNormEpsilon);
  }
  __syncthreads();
  for (int column = lane; column < 2048; column += blockDim.x) {
    output[base + column] = __float2bfloat16_rn(
        __bfloat162float(input[base + column]) * inverse_rms * weight[column]);
  }
}
