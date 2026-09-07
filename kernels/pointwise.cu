#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

template <typename Input, typename Output, typename Binary>
__device__ __forceinline__ void BinaryGridStride(
    const Input* lhs, const Input* rhs, Output* output, std::uint64_t numel,
    Binary binary) {
  const std::uint64_t first =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index = first; index < numel; index += stride) {
    output[index] = binary(lhs[index], rhs[index]);
  }
}

struct AddF32 {
  __device__ __forceinline__ float operator()(float lhs, float rhs) const {
    return lhs + rhs;
  }
};

struct AddBf16 {
  __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 lhs,
                                                       __nv_bfloat16 rhs) const {
    return __float2bfloat16_rn(__bfloat162float(lhs) + __bfloat162float(rhs));
  }
};

struct AddI32 {
  __device__ __forceinline__ std::int32_t operator()(std::int32_t lhs,
                                                      std::int32_t rhs) const {
    const auto sum = static_cast<std::uint32_t>(lhs) +
                     static_cast<std::uint32_t>(rhs);
    return static_cast<std::int32_t>(sum);
  }
};

struct MulF32 {
  __device__ __forceinline__ float operator()(float lhs, float rhs) const {
    return lhs * rhs;
  }
};

struct MulBf16 {
  __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 lhs,
                                                       __nv_bfloat16 rhs) const {
    return __float2bfloat16_rn(__bfloat162float(lhs) * __bfloat162float(rhs));
  }
};

}  // namespace

extern "C" __global__ void aginfer_add_f32(
    const float* lhs, const float* rhs, float* output, std::uint64_t numel) {
  BinaryGridStride(lhs, rhs, output, numel, AddF32{});
}

extern "C" __global__ void aginfer_add_bf16(
    const __nv_bfloat16* lhs, const __nv_bfloat16* rhs,
    __nv_bfloat16* output, std::uint64_t numel) {
  BinaryGridStride(lhs, rhs, output, numel, AddBf16{});
}

extern "C" __global__ void aginfer_add_i32(
    const std::int32_t* lhs, const std::int32_t* rhs,
    std::int32_t* output, std::uint64_t numel) {
  BinaryGridStride(lhs, rhs, output, numel, AddI32{});
}

extern "C" __global__ void aginfer_mul_f32(
    const float* lhs, const float* rhs, float* output, std::uint64_t numel) {
  BinaryGridStride(lhs, rhs, output, numel, MulF32{});
}

extern "C" __global__ void aginfer_mul_bf16(
    const __nv_bfloat16* lhs, const __nv_bfloat16* rhs,
    __nv_bfloat16* output, std::uint64_t numel) {
  BinaryGridStride(lhs, rhs, output, numel, MulBf16{});
}
extern "C" __global__ void aginfer_rounded_mul_add_bf16(
    const __nv_bfloat16* lhs, const __nv_bfloat16* rhs,
    const __nv_bfloat16* residual, __nv_bfloat16* output, unsigned long long numel) {
  const auto first=static_cast<unsigned long long>(blockIdx.x)*blockDim.x+threadIdx.x;
  const auto stride=static_cast<unsigned long long>(blockDim.x)*gridDim.x;
  for(auto i=first;i<numel;i+=stride) {
    const auto product=__float2bfloat16_rn(__fmul_rn(__bfloat162float(lhs[i]),__bfloat162float(rhs[i])));
    output[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(product),__bfloat162float(residual[i])));
  }
}
