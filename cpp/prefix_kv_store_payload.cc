#include "prefix_kv_store_payload.h"

#include <algorithm>
#include <array>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'K', 'V', 'S', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kVariant = 1;
constexpr std::uint32_t kBf16 = 2;
constexpr std::uint32_t kBhsd = 1;
constexpr std::uint32_t kBshd = 2;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kHeads = 1;
constexpr std::uint32_t kSequence = 968;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGridX = 242;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kAlignment = 16;
constexpr std::uint64_t kTensorBytes = 495616;

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

}  // namespace

const char* PrefixKvStoreSymbol() noexcept {
  return "aginfer_prefix_kv_store_bf16_h1_s968_d256";
}

Status ParsePrefixKvStorePayload(const std::uint8_t* data, std::size_t size,
                                 PrefixKvStorePayloadView* output) {
  if (data == nullptr || output == nullptr ||
      size != kPrefixKvStorePayloadSize) {
    return Error("prefix KV store payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad prefix KV store payload magic");
  }
  if (ReadU16(data + 8) != kPrefixKvStoreSchemaMajor ||
      ReadU16(data + 10) > kPrefixKvStoreSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 16) != kVariant ||
      ReadU32(data + 20) != kBf16 || ReadU32(data + 24) != kBhsd ||
      ReadU32(data + 28) != kBshd || ReadU32(data + 32) != kBhsd ||
      ReadU32(data + 36) != kBhsd || ReadU32(data + 40) != kBatch ||
      ReadU32(data + 44) != kHeads || ReadU32(data + 48) != kSequence ||
      ReadU32(data + 52) != kHeadDim || ReadU32(data + 56) != kGridX ||
      ReadU32(data + 60) != 1 || ReadU32(data + 64) != 1 ||
      ReadU32(data + 68) != kBlockX || ReadU32(data + 72) != 1 ||
      ReadU32(data + 76) != 1 || ReadU32(data + 80) != kAlignment ||
      ReadU32(data + 84) != 0 || ReadU32(data + 88) != 0 ||
      ReadU64(data + 92) != kTensorBytes ||
      ReadU64(data + 100) != kTensorBytes ||
      ReadU64(data + 108) != kTensorBytes ||
      ReadU64(data + 116) != kTensorBytes || ReadU64(data + 124) == 0 ||
      ReadU64(data + 132) != 0 ||
      !std::any_of(data + 140, data + 172,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 172, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("prefix KV store payload is outside the delivered exact variant");
  }

  PrefixKvStorePayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.grid = {kGridX, 1, 1};
  parsed.block = {kBlockX, 1, 1};
  parsed.alignment = kAlignment;
  parsed.key_bytes = kTensorBytes;
  parsed.value_bytes = kTensorBytes;
  parsed.state_key_bytes = kTensorBytes;
  parsed.state_value_bytes = kTensorBytes;
  parsed.module_bytes = ReadU64(data + 124);
  std::copy(data + 140, data + 172, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
