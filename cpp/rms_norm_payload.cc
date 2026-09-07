#include "rms_norm_payload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'R', 'M', 'S', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kF32 = 1;
constexpr std::uint32_t kBf16 = 2;
constexpr std::uint32_t kRowMajor = 1;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kBlockY = 1;
constexpr std::uint32_t kBlockZ = 1;
constexpr std::uint32_t kSharedBytes = 0;
constexpr std::uint32_t kAlignment = 16;
constexpr std::uint32_t kF32Accumulation = 1;
constexpr std::uint32_t kEpsilonBits = 0x358637bdU;

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

bool MatchVariant(RmsNormVariant variant, std::uint32_t input_dtype,
                  std::uint32_t output_dtype, std::uint32_t rows,
                  std::uint32_t width, std::uint32_t grid_x,
                  std::uint64_t input_bytes, std::uint64_t weight_bytes,
                  std::uint64_t output_bytes) noexcept {
  if (variant == RmsNormVariant::kF32Rows50Width1024) {
    return input_dtype == kF32 && output_dtype == kF32 && rows == 50 &&
           width == 1024 && grid_x == 50 && input_bytes == 204800 &&
           weight_bytes == 4096 && output_bytes == 204800;
  }
  if (variant == RmsNormVariant::kBf16Rows968Width2048) {
    return input_dtype == kBf16 && output_dtype == kBf16 && rows == 968 &&
           width == 2048 && grid_x == 968 && input_bytes == 3964928 &&
           weight_bytes == 8192 && output_bytes == 3964928;
  }
  return false;
}

}  // namespace

const char* RmsNormSymbol(RmsNormVariant variant) noexcept {
  switch (variant) {
    case RmsNormVariant::kF32Rows50Width1024:
      return "aginfer_rms_norm_f32_1024";
    case RmsNormVariant::kBf16Rows968Width2048:
      return "aginfer_rms_norm_bf16_2048";
  }
  return nullptr;
}

Status ParseRmsNormPayload(const std::uint8_t* data, std::size_t size,
                           RmsNormPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kRmsNormPayloadSize) {
    return Error("RMSNorm payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad RMSNorm payload magic");
  }
  if (ReadU16(data + 8) != kRmsNormSchemaMajor ||
      ReadU16(data + 10) > kRmsNormSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 24) != kF32 ||
      ReadU32(data + 32) != kRowMajor || ReadU32(data + 36) != kBatch ||
      ReadU32(data + 52) != 1 || ReadU32(data + 56) != 1 ||
      ReadU32(data + 60) != kBlockX || ReadU32(data + 64) != kBlockY ||
      ReadU32(data + 68) != kBlockZ || ReadU32(data + 72) != kSharedBytes ||
      ReadU32(data + 76) != kAlignment || ReadU32(data + 80) != kAlignment ||
      ReadU32(data + 84) != kAlignment ||
      ReadU32(data + 88) != kF32Accumulation ||
      ReadU32(data + 92) != kEpsilonBits ||
      !std::all_of(data + 160, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("RMSNorm payload is outside the delivered exact variants");
  }

  RmsNormPayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.variant = static_cast<RmsNormVariant>(ReadU32(data + 16));
  parsed.rows = ReadU32(data + 40);
  parsed.width = ReadU32(data + 44);
  parsed.grid = {ReadU32(data + 48), 1, 1};
  parsed.block = {kBlockX, kBlockY, kBlockZ};
  parsed.shared_bytes = kSharedBytes;
  parsed.input_alignment = kAlignment;
  parsed.weight_alignment = kAlignment;
  parsed.output_alignment = kAlignment;
  parsed.epsilon = ReadF32(data + 92);
  parsed.input_bytes = ReadU64(data + 96);
  parsed.weight_bytes = ReadU64(data + 104);
  parsed.output_bytes = ReadU64(data + 112);
  parsed.module_bytes = ReadU64(data + 120);
  std::copy(data + 128, data + 160, parsed.module_sha256.begin());
  if (!MatchVariant(parsed.variant, ReadU32(data + 20), ReadU32(data + 28),
                    parsed.rows, parsed.width, parsed.grid[0],
                    parsed.input_bytes, parsed.weight_bytes,
                    parsed.output_bytes) ||
      parsed.module_bytes == 0 ||
      !std::any_of(parsed.module_sha256.begin(), parsed.module_sha256.end(),
                   [](std::uint8_t value) { return value != 0; })) {
    return Error("RMSNorm payload has an unknown variant or module identity");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
