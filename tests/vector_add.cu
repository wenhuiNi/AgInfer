extern "C" __global__ void vector_add_f32(const float* left,
                                            const float* right,
                                            float* output,
                                            unsigned int count) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = left[index] + right[index];
}

