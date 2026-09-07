#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kLayerNormRows = 256;
constexpr int kLayerNormWidth = 1152;
constexpr float kLayerNormEpsilon = 1.0e-6F;

}  // namespace

extern "C" __global__ void aginfer_layer_norm_f32_1152(
    const float* input, const float* weight, const float* bias,
    float* output) {
  __shared__ float reduction[256];
  const int row = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);
  const std::uint64_t row_offset =
      static_cast<std::uint64_t>(row) * kLayerNormWidth;

  float partial_sum = 0.0F;
  for (int column = lane; column < kLayerNormWidth; column += blockDim.x) {
    partial_sum += input[row_offset + column];
  }
  reduction[lane] = partial_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  const float mean = reduction[0] / static_cast<float>(kLayerNormWidth);
  __syncthreads();

  float partial_variance = 0.0F;
  for (int column = lane; column < kLayerNormWidth; column += blockDim.x) {
    const float centered = input[row_offset + column] - mean;
    partial_variance += centered * centered;
  }
  reduction[lane] = partial_variance;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (lane < stride) reduction[lane] += reduction[lane + stride];
    __syncthreads();
  }
  const float inverse_standard_deviation =
      rsqrtf(reduction[0] / static_cast<float>(kLayerNormWidth) +
             kLayerNormEpsilon);

  for (int column = lane; column < kLayerNormWidth; column += blockDim.x) {
    const float normalized =
        (input[row_offset + column] - mean) * inverse_standard_deviation;
    output[row_offset + column] = normalized * weight[column] + bias[column];
  }
}
