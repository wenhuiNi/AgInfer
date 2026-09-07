#include "layer_norm_payload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'L', 'N', 'R', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kVariant = 1;
constexpr std::uint32_t kF32 = 1;
constexpr std::uint32_t kRowMajor = 1;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kRows = 256;
constexpr std::uint32_t kWidth = 1152;
constexpr std::uint32_t kGridX = 256;
constexpr std::uint32_t kGridY = 1;
constexpr std::uint32_t kGridZ = 1;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kBlockY = 1;
constexpr std::uint32_t kBlockZ = 1;
constexpr std::uint32_t kSharedBytes = 0;
constexpr std::uint32_t kAlignment = 16;
constexpr std::uint32_t kAffine = 1;
constexpr std::uint32_t kTwoPassVariance = 1;
constexpr std::uint32_t kEpsilonBits = 0x358637bdU;
constexpr std::uint64_t kTensorBytes = 1179648;
constexpr std::uint64_t kParameterBytes = 4608;

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

float ReadF32(const std::uint8_t* data) noexcept {
  const std::uint32_t bits = ReadU32(data);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

Status ParseLayerNormPayload(const std::uint8_t* data, std::size_t size,
                             LayerNormPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kLayerNormPayloadSize) {
    return Error("LayerNorm payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad LayerNorm payload magic");
  }
  if (ReadU16(data + 8) != kLayerNormSchemaMajor ||
      ReadU16(data + 10) > kLayerNormSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 16) != kVariant ||
      ReadU32(data + 20) != kF32 || ReadU32(data + 24) != kRowMajor ||
      ReadU32(data + 28) != kBatch || ReadU32(data + 32) != kRows ||
      ReadU32(data + 36) != kWidth || ReadU32(data + 40) != kGridX ||
      ReadU32(data + 44) != kGridY || ReadU32(data + 48) != kGridZ ||
      ReadU32(data + 52) != kBlockX || ReadU32(data + 56) != kBlockY ||
      ReadU32(data + 60) != kBlockZ || ReadU32(data + 64) != kSharedBytes ||
      ReadU32(data + 68) != kAlignment || ReadU32(data + 72) != kAlignment ||
      ReadU32(data + 76) != kAlignment || ReadU32(data + 80) != kAffine ||
      ReadU32(data + 84) != kTwoPassVariance ||
      ReadU32(data + 88) != kEpsilonBits ||
      ReadU64(data + 92) != kTensorBytes ||
      ReadU64(data + 100) != kParameterBytes ||
      ReadU64(data + 108) != kParameterBytes ||
      ReadU64(data + 116) != kTensorBytes ||
      !std::all_of(data + 164, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("LayerNorm payload is outside the delivered exact variant");
  }

  LayerNormPayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.rows = kRows;
  parsed.width = kWidth;
  parsed.grid = {kGridX, kGridY, kGridZ};
  parsed.block = {kBlockX, kBlockY, kBlockZ};
  parsed.shared_bytes = kSharedBytes;
  parsed.input_alignment = kAlignment;
  parsed.parameter_alignment = kAlignment;
  parsed.output_alignment = kAlignment;
  parsed.epsilon = ReadF32(data + 88);
  parsed.input_bytes = ReadU64(data + 92);
  parsed.weight_bytes = ReadU64(data + 100);
  parsed.bias_bytes = ReadU64(data + 108);
  parsed.output_bytes = ReadU64(data + 116);
  parsed.module_bytes = ReadU64(data + 124);
  std::copy(data + 132, data + 164, parsed.module_sha256.begin());
  if (parsed.module_bytes == 0 ||
      !std::any_of(parsed.module_sha256.begin(), parsed.module_sha256.end(),
                   [](std::uint8_t value) { return value != 0; })) {
    return Error("LayerNorm payload has invalid module identity");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
