#include <cuda_runtime.h>
#include <cstdint>

// One uint4 copies eight BF16 values without arithmetic or rounding. Every
// partition width is a multiple of eight; each output remains contiguous.
extern "C" __global__ void aginfer_projection_split_bf16(
    const uint4* input, uint4* first, uint4* second, uint4* third,
    std::uint32_t rows, std::uint32_t n0, std::uint32_t n1, std::uint32_t n2) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  n0 /= 8; n1 /= 8; n2 /= 8;
  const auto width = n0 + n1 + n2;
  if (index >= rows * width) return;
  const auto row = index / width, column = index % width;
  const uint4 value = input[index];
  if (column < n0) first[row * n0 + column] = value;
  else if (column < n0 + n1) second[row * n1 + column - n0] = value;
  else third[row * n2 + column - n0 - n1] = value;
}
