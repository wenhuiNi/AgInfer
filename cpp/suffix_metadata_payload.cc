#include "suffix_metadata_payload.h"

#include <algorithm>
#include <array>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'S', 'M', 'D', '1', 0, 0};

Status Error(const std::string& message) {
  return Status(StatusCode::kCorruptPackage, message);
}

std::uint16_t U16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         static_cast<std::uint16_t>(p[1]) << 8;
}
std::uint32_t U32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         static_cast<std::uint32_t>(p[1]) << 8 |
         static_cast<std::uint32_t>(p[2]) << 16 |
         static_cast<std::uint32_t>(p[3]) << 24;
}
std::uint64_t U64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(p[index]) << (index * 8);
  }
  return value;
}

}  // namespace

const char* SuffixMetadataSymbol() noexcept {
  return "aginfer_suffix_metadata_bool_s968_s50";
}

Status ParseSuffixMetadataPayload(const std::uint8_t* data, std::size_t size,
                                  SuffixMetadataPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kSuffixMetadataPayloadSize) {
    return Error("suffix metadata payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data) || U16(data + 8) != 1 ||
      U16(data + 10) != 0 || U32(data + 12) != 120 || U32(data + 16) != 1 ||
      U32(data + 20) != 5 || U32(data + 24) != 5 || U32(data + 28) != 3 ||
      U32(data + 32) != 1 || U32(data + 36) != 1 || U32(data + 40) != 1 ||
      U32(data + 44) != 1 || U32(data + 48) != 968 || U32(data + 52) != 50 ||
      U32(data + 56) != 1018 || U32(data + 60) != 199 ||
      U32(data + 64) != 256 || U32(data + 68) != 1 || U32(data + 72) != 1 ||
      U32(data + 76) != 4 || U32(data + 80) != 0 || U32(data + 84) != 0 ||
      U32(data + 88) != 0 || U64(data + 92) != 968 ||
      U64(data + 100) != 50900 || U64(data + 108) != 200 ||
      U64(data + 116) == 0 || U64(data + 124) != 0 || U64(data + 132) != 0 ||
      !std::any_of(data + 140, data + 172,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 172, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("suffix metadata payload is outside the delivered exact variant");
  }
  SuffixMetadataPayloadView parsed;
  parsed.target_arch = 120;
  parsed.grid = {199, 1, 1};
  parsed.block = {256, 1, 1};
  parsed.pad_alignment = 1;
  parsed.mask_alignment = 1;
  parsed.position_alignment = 4;
  parsed.pad_bytes = 968;
  parsed.mask_bytes = 50900;
  parsed.position_bytes = 200;
  parsed.module_bytes = U64(data + 116);
  std::copy(data + 140, data + 172, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
