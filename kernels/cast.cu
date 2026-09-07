#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

template <typename Input, typename Output, typename Convert>
__device__ __forceinline__ void CastGridStride(const Input* input,
                                               Output* output,
                                               std::uint64_t numel,
                                               Convert convert) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < numel; index += stride) {
    output[index] = convert(input[index]);
  }
}

struct Bf16ToF32 {
  __device__ __forceinline__ float operator()(__nv_bfloat16 value) const {
    return __bfloat162float(value);
  }
};

struct F32ToBf16 {
  __device__ __forceinline__ __nv_bfloat16 operator()(float value) const {
    return __float2bfloat16_rn(value);
  }
};

struct BoolToI32 {
  __device__ __forceinline__ std::int32_t operator()(std::uint8_t value) const {
    return value == 0 ? 0 : 1;
  }
};

}  // namespace

extern "C" __global__ void aginfer_cast_bf16_to_f32(
    const __nv_bfloat16* input, float* output, std::uint64_t numel) {
  CastGridStride(input, output, numel, Bf16ToF32{});
}

extern "C" __global__ void aginfer_cast_f32_to_bf16(
    const float* input, __nv_bfloat16* output, std::uint64_t numel) {
  CastGridStride(input, output, numel, F32ToBf16{});
}

extern "C" __global__ void aginfer_cast_bool_to_i32(
    const std::uint8_t* input, std::int32_t* output, std::uint64_t numel) {
  CastGridStride(input, output, numel, BoolToI32{});
}
