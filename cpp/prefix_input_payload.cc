#include "prefix_input_payload.h"

#include <algorithm>
#include <array>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'P', 'I', 'A', '1', 0, 0};

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

const char* PrefixInputSymbol() noexcept {
  return "aginfer_prefix_input_f32_bf16_s968_d2048";
}

Status ParsePrefixInputPayload(const std::uint8_t* data, std::size_t size,
                               PrefixInputPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kPrefixInputPayloadSize) {
    return Error("prefix input payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data) || U16(data + 8) != 1 ||
      U16(data + 10) != 0 || U32(data + 12) != 120 || U32(data + 16) != 1 ||
      U32(data + 20) != 9 || U32(data + 24) != 3 || U32(data + 28) != 3 ||
      U32(data + 32) != 256 || U32(data + 36) != 200 ||
      U32(data + 40) != 2048 || U32(data + 44) != 257152 ||
      U32(data + 48) != 968 || U32(data + 52) != 7744 ||
      U32(data + 56) != 256 || U32(data + 60) != 4 || U32(data + 64) != 2 ||
      U32(data + 68) != 4 || U32(data + 72) != 1 || U32(data + 76) != 4 ||
      U32(data + 80) != 4 || U32(data + 84) != 0 || U32(data + 88) != 0 ||
      U64(data + 92) != 2097152 || U64(data + 100) != 1053294592 ||
      U64(data + 108) != 800 || U64(data + 116) != 7929856 ||
      U64(data + 124) == 0 || U64(data + 132) != 0 ||
      !std::any_of(data + 140, data + 172,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 172, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("prefix input payload is outside the delivered exact variant");
  }
  PrefixInputPayloadView parsed;
  parsed.target_arch = 120;
  parsed.grid = {7744, 1, 1};
  parsed.block = {256, 1, 1};
  parsed.image_alignment = 4;
  parsed.embedding_alignment = 2;
  parsed.token_alignment = 4;
  parsed.mask_alignment = 1;
  parsed.prefix_alignment = 4;
  parsed.position_alignment = 4;
  parsed.image_bytes = 2097152;
  parsed.embedding_bytes = 1053294592;
  parsed.token_bytes = 800;
  parsed.prefix_bytes = 7929856;
  parsed.module_bytes = U64(data + 124);
  std::copy(data + 140, data + 172, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
