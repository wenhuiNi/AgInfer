#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float GeluTanh(float value) {
  constexpr float kSqrtTwoOverPi = 0.7978845608F;
  constexpr float kCubic = 0.044715F;
  const float cube = value * value * value;
  return 0.5F * value *
         (1.0F + tanhf(kSqrtTwoOverPi * (value + kCubic * cube)));
}

template <typename Input, typename Output, typename ConvertInput,
          typename ConvertOutput>
__device__ __forceinline__ void GeluGridStride(
    const Input* input, Output* output, std::uint64_t numel,
    ConvertInput convert_input, ConvertOutput convert_output) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < numel; index += stride) {
    output[index] = convert_output(GeluTanh(convert_input(input[index])));
  }
}

struct GeluF32Identity {
  __device__ __forceinline__ float operator()(float value) const {
    return value;
  }
};

struct GeluBf16ToF32 {
  __device__ __forceinline__ float operator()(__nv_bfloat16 value) const {
    return __bfloat162float(value);
  }
};

struct GeluF32ToBf16 {
  __device__ __forceinline__ __nv_bfloat16 operator()(float value) const {
    return __float2bfloat16_rn(value);
  }
};

}  // namespace

extern "C" __global__ void aginfer_gelu_tanh_f32(
    const float* input, float* output, std::uint64_t numel) {
  GeluGridStride(input, output, numel, GeluF32Identity{}, GeluF32Identity{});
}

extern "C" __global__ void aginfer_gelu_tanh_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    std::uint64_t numel) {
  GeluGridStride(input, output, numel, GeluBf16ToF32{}, GeluF32ToBf16{});
}

extern "C" __global__ void aginfer_silu_f32(
    const float* input, float* output, std::uint64_t numel) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < numel; index += stride) {
    const float value = input[index];
    output[index] = value / (1.0F + expf(-value));
  }
}
