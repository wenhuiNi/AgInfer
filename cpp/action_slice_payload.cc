#include "action_slice_payload.h"

#include <algorithm>
#include <array>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'S', 'L', 'C', '1', 0, 0};
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

const char* ActionSliceSymbol() noexcept {
  return "aginfer_action_slice_f32_r50_w32_o7";
}

Status ParseActionSlicePayload(const std::uint8_t* data, std::size_t size,
                               ActionSlicePayloadView* output) {
  if (data == nullptr || output == nullptr || size != kActionSlicePayloadSize) {
    return Error("action slice payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data) || U16(data + 8) != 1 ||
      U16(data + 10) != 0 || U32(data + 12) != 120 || U32(data + 16) != 1 ||
      U32(data + 20) != 1 || U32(data + 24) != 3 || U32(data + 28) != 50 ||
      U32(data + 32) != 32 || U32(data + 36) != 7 || U32(data + 40) != 350 ||
      U32(data + 44) != 2 || U32(data + 48) != 256 ||
      U32(data + 52) != 4 || U32(data + 56) != 4 ||
      U32(data + 60) != 1 || U32(data + 64) != 1 ||
      U32(data + 68) != 2 || U32(data + 72) != 0 || U32(data + 76) != 7 ||
      U32(data + 80) != 1 || U32(data + 84) != 0 || U32(data + 88) != 0 ||
      U64(data + 92) != 6400 || U64(data + 100) != 1400 ||
      U64(data + 108) == 0 || U64(data + 116) != 0 ||
      U64(data + 124) != 0 || U64(data + 132) != 0 ||
      !std::any_of(data + 140, data + 172,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 172, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("action slice payload is outside the delivered exact variant");
  }
  ActionSlicePayloadView parsed;
  parsed.target_arch = 120;
  parsed.grid = {2, 1, 1};
  parsed.block = {256, 1, 1};
  parsed.input_alignment = 4;
  parsed.output_alignment = 4;
  parsed.input_bytes = 6400;
  parsed.output_bytes = 1400;
  parsed.module_bytes = U64(data + 108);
  std::copy(data + 140, data + 172, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
