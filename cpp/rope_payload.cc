#include "rope_payload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'R', 'O', 'P', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kBf16 = 2;
constexpr std::uint32_t kI32 = 3;
constexpr std::uint32_t kBshd = 2;
constexpr std::uint32_t kBhsd = 3;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kHalfDim = 128;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kBlockY = 1;
constexpr std::uint32_t kBlockZ = 1;
constexpr std::uint32_t kSharedBytes = 0;
constexpr std::uint32_t kAlignment = 16;
constexpr std::uint32_t kFrequencyBf16 = 2;
constexpr std::uint32_t kSplitHalf = 1;
constexpr std::uint32_t kThetaBits = 0x461c4000U;

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

bool MatchVariant(RopeVariant variant, std::uint32_t heads,
                  std::uint32_t sequence, std::uint32_t grid_x,
                  std::uint64_t input_bytes, std::uint64_t position_bytes,
                  std::uint64_t output_bytes) noexcept {
  if (variant == RopeVariant::kBf16Sequence968Heads8) {
    return heads == 8 && sequence == 968 && grid_x == 3872 &&
           input_bytes == 3964928 && position_bytes == 3872 &&
           output_bytes == 3964928;
  }
  if (variant == RopeVariant::kBf16Sequence968Heads1) {
    return heads == 1 && sequence == 968 && grid_x == 484 &&
           input_bytes == 495616 && position_bytes == 3872 &&
           output_bytes == 495616;
  }
  if (variant == RopeVariant::kBf16Sequence50Heads8) {
    return heads == 8 && sequence == 50 && grid_x == 200 &&
           input_bytes == 204800 && position_bytes == 200 &&
           output_bytes == 204800;
  }
  if (variant == RopeVariant::kBf16Sequence50Heads1) {
    return heads == 1 && sequence == 50 && grid_x == 25 &&
           input_bytes == 25600 && position_bytes == 200 &&
           output_bytes == 25600;
  }
  return false;
}

}  // namespace

const char* RopeSymbol(RopeVariant variant) noexcept {
  switch (variant) {
    case RopeVariant::kBf16Sequence968Heads8:
      return "aginfer_rope_bf16_s968_h8";
    case RopeVariant::kBf16Sequence968Heads1:
      return "aginfer_rope_bf16_s968_h1";
    case RopeVariant::kBf16Sequence50Heads8:
      return "aginfer_rope_bf16_s50_h8";
    case RopeVariant::kBf16Sequence50Heads1:
      return "aginfer_rope_bf16_s50_h1";
  }
  return nullptr;
}

Status ParseRopePayload(const std::uint8_t* data, std::size_t size,
                        RopePayloadView* output) {
  if (data == nullptr || output == nullptr || size != kRopePayloadSize) {
    return Error("RoPE payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad RoPE payload magic");
  }
  if (ReadU16(data + 8) != kRopeSchemaMajor ||
      ReadU16(data + 10) > kRopeSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 20) != kBf16 ||
      ReadU32(data + 24) != kI32 || ReadU32(data + 28) != kBf16 ||
      ReadU32(data + 32) != kBshd || ReadU32(data + 36) != kBhsd ||
      ReadU32(data + 40) != kBatch || ReadU32(data + 52) != kHeadDim ||
      ReadU32(data + 56) != kHalfDim || ReadU32(data + 64) != 1 ||
      ReadU32(data + 68) != 1 || ReadU32(data + 72) != kBlockX ||
      ReadU32(data + 76) != kBlockY || ReadU32(data + 80) != kBlockZ ||
      ReadU32(data + 84) != kSharedBytes ||
      ReadU32(data + 88) != kAlignment ||
      ReadU32(data + 92) != kAlignment ||
      ReadU32(data + 96) != kAlignment ||
      ReadU32(data + 100) != kFrequencyBf16 ||
      ReadU32(data + 104) != kSplitHalf ||
      ReadU32(data + 108) != kThetaBits ||
      !std::all_of(data + 176, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("RoPE payload is outside the delivered exact variants");
  }

  RopePayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.variant = static_cast<RopeVariant>(ReadU32(data + 16));
  parsed.heads = ReadU32(data + 44);
  parsed.sequence = ReadU32(data + 48);
  parsed.head_dim = kHeadDim;
  parsed.grid = {ReadU32(data + 60), 1, 1};
  parsed.block = {kBlockX, kBlockY, kBlockZ};
  parsed.shared_bytes = kSharedBytes;
  parsed.input_alignment = kAlignment;
  parsed.position_alignment = kAlignment;
  parsed.output_alignment = kAlignment;
  parsed.theta = ReadF32(data + 108);
  parsed.input_bytes = ReadU64(data + 112);
  parsed.position_bytes = ReadU64(data + 120);
  parsed.output_bytes = ReadU64(data + 128);
  parsed.module_bytes = ReadU64(data + 136);
  std::copy(data + 144, data + 176, parsed.module_sha256.begin());
  if (!MatchVariant(parsed.variant, parsed.heads, parsed.sequence,
                    parsed.grid[0], parsed.input_bytes, parsed.position_bytes,
                    parsed.output_bytes) ||
      parsed.module_bytes == 0 ||
      !std::any_of(parsed.module_sha256.begin(), parsed.module_sha256.end(),
                   [](std::uint8_t value) { return value != 0; })) {
    return Error("RoPE payload has an unknown variant or module identity");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
