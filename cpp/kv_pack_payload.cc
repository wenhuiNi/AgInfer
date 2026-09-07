#include "kv_pack_payload.h"

#include <algorithm>
#include <array>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'K', 'V', 'P', '1', 0, 0};
constexpr std::uint32_t kTargetArch = 120;
constexpr std::uint32_t kVariant = 1;
constexpr std::uint32_t kBf16 = 2;
constexpr std::uint32_t kBhsd = 1;
constexpr std::uint32_t kBshd = 2;
constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kHeads = 1;
constexpr std::uint32_t kPrefixSequence = 968;
constexpr std::uint32_t kCurrentSequence = 50;
constexpr std::uint32_t kTotalSequence = 1018;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGridX = 255;
constexpr std::uint32_t kBlockX = 256;
constexpr std::uint32_t kAlignment = 16;
constexpr std::uint64_t kPrefixBytes = 495616;
constexpr std::uint64_t kCurrentBytes = 25600;
constexpr std::uint64_t kPackedBytes = 521216;

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

const char* KvPackSymbol() noexcept {
  return "aginfer_kv_pack_bf16_h1_s968_s50_d256";
}

Status ParseKvPackPayload(const std::uint8_t* data, std::size_t size,
                          KvPackPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kKvPackPayloadSize) {
    return Error("KV pack payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad KV pack payload magic");
  }
  if (ReadU16(data + 8) != kKvPackSchemaMajor ||
      ReadU16(data + 10) > kKvPackSchemaMinor ||
      ReadU32(data + 12) != kTargetArch || ReadU32(data + 16) != kVariant ||
      ReadU32(data + 20) != kBf16 || ReadU32(data + 24) != kBhsd ||
      ReadU32(data + 28) != kBhsd || ReadU32(data + 32) != kBshd ||
      ReadU32(data + 36) != kBhsd || ReadU32(data + 40) != kBatch ||
      ReadU32(data + 44) != kHeads ||
      ReadU32(data + 48) != kPrefixSequence ||
      ReadU32(data + 52) != kCurrentSequence ||
      ReadU32(data + 56) != kTotalSequence ||
      ReadU32(data + 60) != kHeadDim || ReadU32(data + 64) != kGridX ||
      ReadU32(data + 68) != 1 || ReadU32(data + 72) != 1 ||
      ReadU32(data + 76) != kBlockX || ReadU32(data + 80) != 1 ||
      ReadU32(data + 84) != 1 || ReadU32(data + 88) != kAlignment ||
      ReadU64(data + 92) != kPrefixBytes ||
      ReadU64(data + 100) != kCurrentBytes ||
      ReadU64(data + 108) != kCurrentBytes ||
      ReadU64(data + 116) != kPackedBytes ||
      ReadU64(data + 124) != kPackedBytes || ReadU64(data + 132) == 0 ||
      !std::any_of(data + 140, data + 172,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 172, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("KV pack payload is outside the delivered exact variant");
  }

  KvPackPayloadView parsed;
  parsed.target_arch = kTargetArch;
  parsed.grid = {kGridX, 1, 1};
  parsed.block = {kBlockX, 1, 1};
  parsed.alignment = kAlignment;
  parsed.prefix_bytes = kPrefixBytes;
  parsed.current_k_bytes = kCurrentBytes;
  parsed.current_v_bytes = kCurrentBytes;
  parsed.packed_k_bytes = kPackedBytes;
  parsed.packed_v_bytes = kPackedBytes;
  parsed.module_bytes = ReadU64(data + 132);
  std::copy(data + 140, data + 172, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
