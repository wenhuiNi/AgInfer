#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kCudaKernelPayloadSize = 128;
inline constexpr std::uint16_t kCudaKernelSchemaMajor = 1;
inline constexpr std::uint16_t kCudaKernelSchemaMinor = 0;

enum class CudaKernelId : std::uint32_t {
  kCastBf16ToF32 = 1,
  kCastF32ToBf16 = 2,
  kCastBoolToI32 = 3,
  kAddF32 = 4,
  kAddBf16 = 5,
  kAddI32 = 6,
  kMulF32 = 7,
  kMulBf16 = 8,
  kGeluF32 = 9,
  kGeluBf16 = 10,
  kSiluF32 = 11,
};

enum class CudaKernelLaunchAbi : std::uint32_t {
  kUnaryPointersNumel = 1,
  kBinaryPointersNumel = 2,
};

enum class CudaKernelDType : std::uint32_t {
  kF32 = 1,
  kBf16 = 2,
  kI32 = 3,
  kBool = 4,
};

struct CudaKernelPayloadView {
  std::uint32_t target_arch = 0;
  CudaKernelId kernel_id = CudaKernelId::kCastBf16ToF32;
  CudaKernelDType input_dtype = CudaKernelDType::kBf16;
  CudaKernelDType output_dtype = CudaKernelDType::kF32;
  std::uint32_t grid_x = 0;
  std::uint32_t block_x = 0;
  std::uint32_t shared_bytes = 0;
  std::uint64_t numel = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
  std::uint32_t input_alignment = 0;
  std::uint32_t output_alignment = 0;
  CudaKernelLaunchAbi launch_abi = CudaKernelLaunchAbi::kUnaryPointersNumel;
};

Status ParseCudaKernelPayload(const std::uint8_t* data, std::size_t size,
                              CudaKernelPayloadView* output);
const char* CudaKernelSymbol(CudaKernelId kernel_id) noexcept;

}  // namespace aginfer::internal
