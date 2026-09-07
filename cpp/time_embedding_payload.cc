#include "time_embedding_payload.h"

#include <algorithm>
#include <array>
#include <bit>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'T', 'E', 'M', '1', 0, 0};

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

const char* TimeEmbeddingSymbol() noexcept {
  return "aginfer_time_embedding_f32_d1024";
}

Status ParseTimeEmbeddingPayload(const std::uint8_t* data, std::size_t size,
                                 TimeEmbeddingPayloadView* output) {
  if (data == nullptr || output == nullptr || size != kTimeEmbeddingPayloadSize) {
    return Error("time embedding payload must have its fixed size");
  }
  constexpr std::uint64_t kMinimumPeriodBits =
      std::bit_cast<std::uint64_t>(0.004);
  constexpr std::uint64_t kMaximumPeriodBits =
      std::bit_cast<std::uint64_t>(4.0);
  if (!std::equal(kMagic.begin(), kMagic.end(), data) || U16(data + 8) != 1 ||
      U16(data + 10) != 0 || U32(data + 12) != 120 || U32(data + 16) != 1 ||
      U32(data + 20) != 1 || U32(data + 24) != 1 ||
      U32(data + 28) != 1024 || U32(data + 32) != 512 ||
      U32(data + 36) != 2 || U32(data + 40) != 256 ||
      U32(data + 44) != 4 || U32(data + 48) != 4 ||
      U32(data + 52) != 1 || U32(data + 56) != 1 ||
      U32(data + 60) != 2 || U32(data + 64) != 1 ||
      U32(data + 68) != 1 || U32(data + 72) != 1 ||
      U32(data + 76) != 0 || U32(data + 80) != 0 ||
      U32(data + 84) != 0 || U32(data + 88) != 0 ||
      U64(data + 92) != 4 || U64(data + 100) != 4096 ||
      U64(data + 108) == 0 || U64(data + 116) != kMinimumPeriodBits ||
      U64(data + 124) != kMaximumPeriodBits || U64(data + 132) != 0 ||
      !std::any_of(data + 140, data + 172,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 172, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("time embedding payload is outside the delivered exact variant");
  }
  TimeEmbeddingPayloadView parsed;
  parsed.target_arch = 120;
  parsed.grid = {2, 1, 1};
  parsed.block = {256, 1, 1};
  parsed.input_alignment = 4;
  parsed.output_alignment = 4;
  parsed.input_bytes = 4;
  parsed.output_bytes = 4096;
  parsed.module_bytes = U64(data + 108);
  std::copy(data + 140, data + 172, parsed.module_sha256.begin());
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
