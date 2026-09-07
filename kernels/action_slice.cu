#include <cuda_runtime.h>

#include <cstdint>

extern "C" __global__ void aginfer_action_slice_f32_r50_w32_o7(
    const float* input, float* output) {
  constexpr std::uint32_t kOutputWidth = 7;
  constexpr std::uint32_t kInputWidth = 32;
  constexpr std::uint32_t kOutputElements = 350;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kOutputElements) return;
  const std::uint32_t row = index / kOutputWidth;
  const std::uint32_t column = index - row * kOutputWidth;
  output[index] = input[row * kInputWidth + column];
}
