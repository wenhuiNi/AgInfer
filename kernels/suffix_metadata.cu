#include <cuda_runtime.h>

extern "C" __global__ void aginfer_suffix_metadata_bool_s968_s50(
    const bool* prefix_pad, bool* attention_mask, int* positions) {
  __shared__ int sums[256];
  if (blockIdx.x == 0) {
    int local_sum = 0;
    for (int index = threadIdx.x; index < 968; index += blockDim.x) {
      local_sum += prefix_pad[index] ? 1 : 0;
    }
    sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
      __syncthreads();
    }
    if (threadIdx.x < 50) positions[threadIdx.x] = sums[0] + threadIdx.x;
  }
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < 50 * 1018; index += gridDim.x * blockDim.x) {
    const int column = index % 1018;
    attention_mask[index] = column < 968 ? prefix_pad[column] : true;
  }
}
