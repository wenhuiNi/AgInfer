#include "cublaslt_payload.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'L', 'T', 'M', 'M', '1', 0};
constexpr std::uint32_t kComputeF32 = 1;
constexpr std::uint32_t kScaleF32 = 1;
constexpr std::uint32_t kBiasFlag = 1;
constexpr std::uint32_t kBiasEpilogue = 1;
constexpr std::uint32_t kOperationT = 1;
constexpr std::uint32_t kOperationN = 0;
constexpr std::uint32_t kOrderColumn = 0;
constexpr std::uint32_t kMaxPointerAlignment = 1U << 20;

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
  return alignment != 0 && alignment <= kMaxPointerAlignment &&
         (alignment & (alignment - 1)) == 0;
}

bool TensorBytesValid(std::uint64_t rows, std::uint64_t columns,
                      std::uint64_t item_bytes) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return rows != 0 && columns != 0 && rows <= maximum / columns &&
         rows * columns <= maximum / item_bytes;
}

}  // namespace

Status ParseCublasLtLinearPayload(const std::uint8_t* data, std::size_t size,
                                  CublasLtLinearPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kCublasLtLinearPayloadSize) {
    return Error("cuBLASLt linear payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad cuBLASLt linear payload magic");
  }
  if (ReadU16(data + 8) != kCublasLtLinearSchemaMajor ||
      ReadU16(data + 10) > kCublasLtLinearSchemaMinor) {
    return Error("unsupported cuBLASLt linear payload schema");
  }
  if (ReadU32(data + 12) != kCublasLtLinearPayloadSize) {
    return Error("cuBLASLt linear payload declares the wrong size");
  }

  CublasLtLinearPayloadView parsed;
  parsed.target_arch = ReadU32(data + 16);
  parsed.cublaslt_version = ReadU32(data + 20);
  const std::uint32_t raw_dtype = ReadU32(data + 24);
  if (!KnownArch(parsed.target_arch) || parsed.cublaslt_version == 0 ||
      (raw_dtype != static_cast<std::uint32_t>(CublasLtDType::kF32) &&
       raw_dtype != static_cast<std::uint32_t>(CublasLtDType::kBf16))) {
    return Error("cuBLASLt payload has an unknown target, version, or dtype");
  }
  parsed.dtype = static_cast<CublasLtDType>(raw_dtype);
  parsed.compute_mode=ReadU32(data+28);
  if ((parsed.compute_mode!=kComputeF32 && parsed.compute_mode!=2) ||
      (parsed.compute_mode==2 && (parsed.dtype!=CublasLtDType::kF32 || ReadU16(data+10)!=1)) ||
      ReadU32(data + 32) != kScaleF32 ||
      ReadU32(data + 36) != kBiasFlag ||
      ReadU32(data + 40) != kBiasEpilogue ||
      ReadU32(data + 44) != kOperationT ||
      ReadU32(data + 48) != kOperationN ||
      ReadU32(data + 52) != kOrderColumn ||
      ReadU32(data + 56) != kOrderColumn ||
      ReadU32(data + 60) != kOrderColumn ||
      ReadU32(data + 64) != kOrderColumn ||
      !std::all_of(data + 184, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("cuBLASLt payload has invalid fixed layout or reserved fields");
  }

  parsed.m = ReadU64(data + 68);
  parsed.n = ReadU64(data + 76);
  parsed.k = ReadU64(data + 84);
  parsed.lda = ReadU64(data + 92);
  parsed.ldb = ReadU64(data + 100);
  parsed.ldc = ReadU64(data + 108);
  parsed.ldd = ReadU64(data + 116);
  parsed.workspace_bytes = ReadU64(data + 124);
  const std::uint64_t item_bytes =
      parsed.dtype == CublasLtDType::kF32 ? 4 : 2;
  if (parsed.lda != parsed.k || parsed.ldb != parsed.k ||
      parsed.ldc != parsed.n || parsed.ldd != parsed.n ||
      !TensorBytesValid(parsed.m, parsed.k, item_bytes) ||
      !TensorBytesValid(parsed.n, parsed.k, item_bytes) ||
      !TensorBytesValid(parsed.m, parsed.n, item_bytes)) {
    return Error("cuBLASLt payload has invalid matrix dimensions or strides");
  }

  const std::uint32_t raw_algorithm_id = ReadU32(data + 132);
  const std::uint32_t raw_split_k = ReadU32(data + 140);
  if (raw_algorithm_id >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
      raw_split_k >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    return Error("cuBLASLt payload has an invalid algorithm ID or split-K");
  }
  parsed.algorithm.algorithm_id = static_cast<std::int32_t>(raw_algorithm_id);
  parsed.algorithm.tile_id = ReadU32(data + 136);
  parsed.algorithm.split_k = static_cast<std::int32_t>(raw_split_k);
  parsed.algorithm.reduction_scheme = ReadU32(data + 144);
  parsed.algorithm.cta_swizzling = ReadU32(data + 148);
  parsed.algorithm.custom_option = ReadU32(data + 152);
  parsed.algorithm.stages_id = ReadU32(data + 156);
  parsed.algorithm.inner_shape_id = ReadU16(data + 160);
  parsed.algorithm.cluster_shape_id = ReadU16(data + 162);
  parsed.x_alignment = ReadU32(data + 164);
  parsed.weight_alignment = ReadU32(data + 168);
  parsed.bias_alignment = ReadU32(data + 172);
  parsed.output_alignment = ReadU32(data + 176);
  parsed.workspace_alignment = ReadU32(data + 180);
  if (!ValidAlignment(parsed.x_alignment) ||
      !ValidAlignment(parsed.weight_alignment) ||
      !ValidAlignment(parsed.bias_alignment) ||
      !ValidAlignment(parsed.output_alignment) ||
      !ValidAlignment(parsed.workspace_alignment)) {
    return Error("cuBLASLt payload has an invalid pointer alignment");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
