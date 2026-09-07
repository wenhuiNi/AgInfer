#include "adaptive_rms_norm_payload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'A', 'R', 'M', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kVariant = 1;
constexpr std::uint32_t kF32 = 1;
constexpr std::uint32_t kBf16 = 2;
constexpr std::uint32_t kRowMajor = 1;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kRows = 50;
constexpr std::uint32_t kWidth = 1024;
constexpr std::uint32_t kModulationWidth = 3072;
constexpr std::uint32_t kOutputs = 2;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kAlignment = 16;
constexpr std::uint32_t kF32Accumulation = 1;
constexpr std::uint32_t kEpsilonBits = 0x358637bdU;
constexpr std::uint64_t kTensorBytes = 102400;
constexpr std::uint64_t kModulationBytes = 12288;

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

const char* AdaptiveRmsNormSymbol() noexcept {
  return "aginfer_adaptive_rms_norm_bf16_f32_1024";
}

Status ParseAdaptiveRmsNormPayload(const std::uint8_t* data, std::size_t size,
                                   AdaptiveRmsNormPayloadView* output) {
  if (data == nullptr || output == nullptr ||
      size != kAdaptiveRmsNormPayloadSize) {
    return Error("adaptive RMSNorm payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad adaptive RMSNorm payload magic");
  }
  if (ReadU16(data + 8) != kAdaptiveRmsNormSchemaMajor ||
      ReadU16(data + 10) > kAdaptiveRmsNormSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 16) != kVariant ||
      ReadU32(data + 20) != kBf16 || ReadU32(data + 24) != kF32 ||
      ReadU32(data + 28) != kBf16 || ReadU32(data + 32) != kBf16 ||
      ReadU32(data + 36) != kRowMajor || ReadU32(data + 40) != kBatch ||
      ReadU32(data + 44) != kRows || ReadU32(data + 48) != kWidth ||
      ReadU32(data + 52) != kModulationWidth ||
      ReadU32(data + 56) != kOutputs || ReadU32(data + 60) != kRows ||
      ReadU32(data + 64) != 1 || ReadU32(data + 68) != 1 ||
      ReadU32(data + 72) != kBlockX || ReadU32(data + 76) != 1 ||
      ReadU32(data + 80) != 1 || ReadU32(data + 84) != 0 ||
      ReadU32(data + 88) != kAlignment ||
      ReadU32(data + 92) != kAlignment ||
      ReadU32(data + 96) != kAlignment ||
      ReadU32(data + 100) != kAlignment ||
      ReadU32(data + 104) != kF32Accumulation ||
      ReadU32(data + 108) != kEpsilonBits ||
      ReadU64(data + 112) != kTensorBytes ||
      ReadU64(data + 120) != kModulationBytes ||
      ReadU64(data + 128) != kTensorBytes ||
      ReadU64(data + 136) != kTensorBytes || ReadU64(data + 144) == 0 ||
      !std::any_of(data + 152, data + 184,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 184, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("adaptive RMSNorm payload is outside the delivered exact variant");
  }

  AdaptiveRmsNormPayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.rows = kRows;
  parsed.width = kWidth;
  parsed.modulation_width = kModulationWidth;
  parsed.grid = {kRows, 1, 1};
  parsed.block = {kBlockX, 1, 1};
  parsed.shared_bytes = 0;
  parsed.hidden_alignment = kAlignment;
  parsed.modulation_alignment = kAlignment;
  parsed.normalized_alignment = kAlignment;
  parsed.gate_alignment = kAlignment;
  parsed.epsilon = ReadF32(data + 108);
  parsed.hidden_bytes = kTensorBytes;
  parsed.modulation_bytes = kModulationBytes;
  parsed.normalized_bytes = kTensorBytes;
  parsed.gate_bytes = kTensorBytes;
  parsed.module_bytes = ReadU64(data + 144);
  std::copy(data + 152, data + 184, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
