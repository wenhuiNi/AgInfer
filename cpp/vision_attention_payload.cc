#include "vision_attention_payload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'V', 'A', 'T', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kVariant = 1;
constexpr std::uint32_t kF32 = 1;
constexpr std::uint32_t kBool = 4;
constexpr std::uint32_t kBshd = 1;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kQueryLength = 256;
constexpr std::uint32_t kKeyLength = 256;
constexpr std::uint32_t kNumHeads = 16;
constexpr std::uint32_t kHeadDim = 72;
constexpr std::uint32_t kGridX = 256;
constexpr std::uint32_t kGridY = 16;
constexpr std::uint32_t kGridZ = 1;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kBlockY = 1;
constexpr std::uint32_t kBlockZ = 1;
constexpr std::uint32_t kSharedBytes = 0;
constexpr std::uint32_t kFiniteDtypeMinMask = 1;
constexpr std::uint32_t kScaleBits = 0x3df15befU;
constexpr std::uint64_t kTensorBytes = 1179648;
constexpr std::uint64_t kMaskBytes = 1;

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

Status ParseVisionAttentionPayload(const std::uint8_t* data, std::size_t size,
                                   VisionAttentionPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kVisionAttentionPayloadSize) {
    return Error("vision attention payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad vision attention payload magic");
  }
  if (ReadU16(data + 8) != kVisionAttentionSchemaMajor ||
      ReadU16(data + 10) > kVisionAttentionSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 16) != kVariant ||
      ReadU32(data + 20) != kF32 || ReadU32(data + 24) != kF32 ||
      ReadU32(data + 28) != kF32 || ReadU32(data + 32) != kBool ||
      ReadU32(data + 36) != kBshd || ReadU32(data + 40) != kBshd ||
      ReadU32(data + 44) != kBatch ||
      ReadU32(data + 48) != kQueryLength ||
      ReadU32(data + 52) != kKeyLength ||
      ReadU32(data + 56) != kNumHeads || ReadU32(data + 60) != kHeadDim ||
      ReadU32(data + 64) != kGridX || ReadU32(data + 68) != kGridY ||
      ReadU32(data + 72) != kGridZ || ReadU32(data + 76) != kBlockX ||
      ReadU32(data + 80) != kBlockY || ReadU32(data + 84) != kBlockZ ||
      ReadU32(data + 88) != kSharedBytes ||
      ReadU32(data + 92) != kFiniteDtypeMinMask ||
      ReadU32(data + 96) != kScaleBits || ReadU64(data + 100) != kTensorBytes ||
      ReadU64(data + 108) != kTensorBytes ||
      ReadU64(data + 116) != kTensorBytes || ReadU64(data + 124) != kMaskBytes ||
      ReadU64(data + 132) != kTensorBytes ||
      !std::all_of(data + 180, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("vision attention payload is outside the delivered exact variant");
  }

  VisionAttentionPayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.query_length = kQueryLength;
  parsed.key_length = kKeyLength;
  parsed.num_heads = kNumHeads;
  parsed.head_dim = kHeadDim;
  parsed.grid = {kGridX, kGridY, kGridZ};
  parsed.block = {kBlockX, kBlockY, kBlockZ};
  parsed.shared_bytes = kSharedBytes;
  parsed.scale = ReadF32(data + 96);
  parsed.query_bytes = ReadU64(data + 100);
  parsed.key_bytes = ReadU64(data + 108);
  parsed.value_bytes = ReadU64(data + 116);
  parsed.mask_bytes = ReadU64(data + 124);
  parsed.output_bytes = ReadU64(data + 132);
  parsed.module_bytes = ReadU64(data + 140);
  std::copy(data + 148, data + 180, parsed.module_sha256.begin());
  if (parsed.module_bytes == 0 ||
      !std::any_of(parsed.module_sha256.begin(), parsed.module_sha256.end(),
                   [](std::uint8_t value) { return value != 0; })) {
    return Error("vision attention payload has invalid module identity");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
