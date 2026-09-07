#include "cuda_kernel_payload.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'C', 'U', 'K', 'R', '1', 0};
constexpr std::uint32_t kContiguousFlag = 1;
constexpr std::uint32_t kCubinModuleFormat = 1;
constexpr std::uint32_t kMaximumGridX = 65535;
constexpr std::uint32_t kMaximumBlockX = 1024;
constexpr std::uint32_t kMaximumSharedBytes = 96 * 1024;
constexpr std::uint32_t kMaximumAlignment = 1U << 20;

Status Error(const std::string& message) {
  return Status(StatusCode::kCorruptPackage, message);
}

std::uint16_t ReadU16(const std::uint8_t* data) noexcept {
  return static_cast<std::uint16_t>(data[0]) |
         static_cast<std::uint16_t>(data[1]) << 8;
}

std::uint32_t ReadU32(const std::uint8_t* data) noexcept {
  return static_cast<std::uint32_t>(data[0]) |
         static_cast<std::uint32_t>(data[1]) << 8 |
         static_cast<std::uint32_t>(data[2]) << 16 |
         static_cast<std::uint32_t>(data[3]) << 24;
}

std::uint64_t ReadU64(const std::uint8_t* data) noexcept {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(data[index]) << (index * 8);
  }
  return value;
}

bool KnownArch(std::uint32_t arch) noexcept {
  return arch == 89 || arch == 110 || arch == 120;
}

bool ValidAlignment(std::uint32_t alignment) noexcept {
  return alignment != 0 && alignment <= kMaximumAlignment &&
         (alignment & (alignment - 1)) == 0;
}

std::uint32_t DTypeBytes(CudaKernelDType dtype) noexcept {
  switch (dtype) {
    case CudaKernelDType::kF32:
    case CudaKernelDType::kI32:
      return 4;
    case CudaKernelDType::kBf16:
      return 2;
    case CudaKernelDType::kBool:
      return 1;
  }
  return 0;
}

bool KnownKernelContract(CudaKernelId kernel_id, CudaKernelDType input,
                         CudaKernelDType output) noexcept {
  return (kernel_id == CudaKernelId::kCastBf16ToF32 &&
          input == CudaKernelDType::kBf16 &&
          output == CudaKernelDType::kF32) ||
         (kernel_id == CudaKernelId::kCastF32ToBf16 &&
          input == CudaKernelDType::kF32 &&
          output == CudaKernelDType::kBf16) ||
         (kernel_id == CudaKernelId::kCastBoolToI32 &&
          input == CudaKernelDType::kBool &&
          output == CudaKernelDType::kI32) ||
         (kernel_id == CudaKernelId::kAddF32 &&
          input == CudaKernelDType::kF32 &&
          output == CudaKernelDType::kF32) ||
         (kernel_id == CudaKernelId::kAddBf16 &&
          input == CudaKernelDType::kBf16 &&
          output == CudaKernelDType::kBf16) ||
         (kernel_id == CudaKernelId::kAddI32 &&
          input == CudaKernelDType::kI32 &&
          output == CudaKernelDType::kI32) ||
         (kernel_id == CudaKernelId::kMulF32 &&
          input == CudaKernelDType::kF32 &&
          output == CudaKernelDType::kF32) ||
         (kernel_id == CudaKernelId::kMulBf16 &&
          input == CudaKernelDType::kBf16 &&
          output == CudaKernelDType::kBf16) ||
         (kernel_id == CudaKernelId::kGeluF32 &&
          input == CudaKernelDType::kF32 &&
          output == CudaKernelDType::kF32) ||
         (kernel_id == CudaKernelId::kGeluBf16 &&
          input == CudaKernelDType::kBf16 &&
          output == CudaKernelDType::kBf16) ||
         (kernel_id == CudaKernelId::kSiluF32 &&
          input == CudaKernelDType::kF32 &&
          output == CudaKernelDType::kF32);
}

bool KnownElementCount(CudaKernelId kernel_id, std::uint64_t numel) noexcept {
  if (kernel_id == CudaKernelId::kGeluF32) return numel == 1101824;
  if (kernel_id == CudaKernelId::kGeluBf16) {
    return numel == 204800 || numel == 15859712;
  }
  if (kernel_id == CudaKernelId::kSiluF32) return numel == 1024;
  return true;
}

CudaKernelLaunchAbi ExpectedLaunchAbi(CudaKernelId kernel_id) noexcept {
  switch (kernel_id) {
    case CudaKernelId::kCastBf16ToF32:
    case CudaKernelId::kCastF32ToBf16:
    case CudaKernelId::kCastBoolToI32:
    case CudaKernelId::kGeluF32:
    case CudaKernelId::kGeluBf16:
    case CudaKernelId::kSiluF32:
      return CudaKernelLaunchAbi::kUnaryPointersNumel;
    case CudaKernelId::kAddF32:
    case CudaKernelId::kAddBf16:
    case CudaKernelId::kAddI32:
    case CudaKernelId::kMulF32:
    case CudaKernelId::kMulBf16:
      return CudaKernelLaunchAbi::kBinaryPointersNumel;
  }
  return static_cast<CudaKernelLaunchAbi>(0);
}

}  // namespace

const char* CudaKernelSymbol(CudaKernelId kernel_id) noexcept {
  switch (kernel_id) {
    case CudaKernelId::kCastBf16ToF32:
      return "aginfer_cast_bf16_to_f32";
    case CudaKernelId::kCastF32ToBf16:
      return "aginfer_cast_f32_to_bf16";
    case CudaKernelId::kCastBoolToI32:
      return "aginfer_cast_bool_to_i32";
    case CudaKernelId::kAddF32:
      return "aginfer_add_f32";
    case CudaKernelId::kAddBf16:
      return "aginfer_add_bf16";
    case CudaKernelId::kAddI32:
      return "aginfer_add_i32";
    case CudaKernelId::kMulF32:
      return "aginfer_mul_f32";
    case CudaKernelId::kMulBf16:
      return "aginfer_mul_bf16";
    case CudaKernelId::kGeluF32:
      return "aginfer_gelu_tanh_f32";
    case CudaKernelId::kGeluBf16:
      return "aginfer_gelu_tanh_bf16";
    case CudaKernelId::kSiluF32:
      return "aginfer_silu_f32";
  }
  return nullptr;
}

Status ParseCudaKernelPayload(const std::uint8_t* data, std::size_t size,
                              CudaKernelPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kCudaKernelPayloadSize) {
    return Error("CUDA kernel payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad CUDA kernel payload magic");
  }
  if (ReadU16(data + 8) != kCudaKernelSchemaMajor ||
      ReadU16(data + 10) > kCudaKernelSchemaMinor ||
      ReadU32(data + 12) != kCudaKernelPayloadSize) {
    return Error("unsupported CUDA kernel payload schema or size");
  }

  CudaKernelPayloadView parsed;
  parsed.target_arch = ReadU32(data + 16);
  parsed.kernel_id = static_cast<CudaKernelId>(ReadU32(data + 20));
  parsed.input_dtype = static_cast<CudaKernelDType>(ReadU32(data + 24));
  parsed.output_dtype = static_cast<CudaKernelDType>(ReadU32(data + 28));
  parsed.grid_x = ReadU32(data + 36);
  parsed.block_x = ReadU32(data + 40);
  parsed.shared_bytes = ReadU32(data + 44);
  parsed.numel = ReadU64(data + 48);
  parsed.module_bytes = ReadU64(data + 56);
  std::copy(data + 64, data + 96, parsed.module_sha256.begin());
  parsed.input_alignment = ReadU32(data + 96);
  parsed.output_alignment = ReadU32(data + 100);
  parsed.launch_abi = static_cast<CudaKernelLaunchAbi>(ReadU32(data + 108));

  if (!KnownArch(parsed.target_arch) ||
      ReadU32(data + 32) != kContiguousFlag ||
      ReadU32(data + 104) != kCubinModuleFormat ||
      !std::all_of(data + 112, data + 128,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("CUDA kernel payload has invalid fixed fields");
  }
  if (!KnownKernelContract(parsed.kernel_id, parsed.input_dtype,
                           parsed.output_dtype) ||
      parsed.launch_abi != ExpectedLaunchAbi(parsed.kernel_id)) {
    return Error("CUDA kernel payload has an unknown kernel contract");
  }
  if (parsed.numel == 0 || !KnownElementCount(parsed.kernel_id, parsed.numel) ||
      parsed.module_bytes == 0 ||
      !std::any_of(parsed.module_sha256.begin(), parsed.module_sha256.end(),
                   [](std::uint8_t value) { return value != 0; }) ||
      parsed.grid_x == 0 || parsed.grid_x > kMaximumGridX ||
      parsed.block_x == 0 || parsed.block_x > kMaximumBlockX ||
      parsed.shared_bytes > kMaximumSharedBytes ||
      !ValidAlignment(parsed.input_alignment) ||
      !ValidAlignment(parsed.output_alignment)) {
    return Error("CUDA kernel payload has invalid module, launch, or alignment fields");
  }
  const std::uint64_t minimum_grid =
      (parsed.numel + parsed.block_x - 1) / parsed.block_x;
  const std::uint64_t expected_grid =
      std::min<std::uint64_t>(minimum_grid, kMaximumGridX);
  if (parsed.grid_x != expected_grid || parsed.shared_bytes != 0) {
    return Error("CUDA kernel payload launch geometry is not canonical");
  }
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (parsed.numel > maximum / DTypeBytes(parsed.input_dtype) ||
      parsed.numel > maximum / DTypeBytes(parsed.output_dtype)) {
    return Error("CUDA kernel payload tensor byte size overflows uint64");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
