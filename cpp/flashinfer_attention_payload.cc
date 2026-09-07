#include "flashinfer_attention_payload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'A', 'I', 'F', 'I', 'A', 'T', '1', 0};
constexpr std::array<std::uint8_t, 20> kFlashInferCommit = {
    0x69, 0xff, 0x11, 0xfc, 0x49, 0x54, 0x39, 0x6d, 0x98, 0x32,
    0x66, 0x56, 0xdc, 0x85, 0xde, 0xbd, 0x22, 0x23, 0xf6, 0x37};
constexpr std::array<std::uint8_t, 20> kCcclCommit = {
    0x87, 0x68, 0x67, 0x68, 0x4f, 0x7f, 0xac, 0x13, 0x0e, 0x0f,
    0x59, 0x11, 0x23, 0x6e, 0x0a, 0x92, 0xa9, 0x70, 0xd4, 0xfd};

Status Invalid(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}

std::uint16_t ReadU16(const std::uint8_t* data, std::size_t offset) {
  return static_cast<std::uint16_t>(data[offset]) |
         (static_cast<std::uint16_t>(data[offset + 1]) << 8);
}

std::uint32_t ReadU32(const std::uint8_t* data, std::size_t offset) {
  std::uint32_t value = 0;
  for (std::size_t index = 0; index < 4; ++index) {
    value |= static_cast<std::uint32_t>(data[offset + index]) << (8 * index);
  }
  return value;
}

std::uint64_t ReadU64(const std::uint8_t* data, std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(data[offset + index]) << (8 * index);
  }
  return value;
}

}  // namespace

Status ParseFlashInferAttentionPayload(
    const std::uint8_t* data, std::size_t size,
    FlashInferAttentionPayloadView* output) {
  if (output == nullptr) return Invalid("FlashInfer attention payload output is null");
  *output = {};
  if (data == nullptr || size != kFlashInferAttentionPayloadSize) {
    return Invalid("FlashInfer attention payload must have its exact fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Invalid("FlashInfer attention payload magic is invalid");
  }
  if (ReadU16(data, 8) != kFlashInferAttentionSchemaMajor ||
      ReadU16(data, 10) > kFlashInferAttentionSchemaMinor) {
    return Invalid("FlashInfer attention payload schema is unsupported");
  }
  if (std::any_of(data + 176, data + size,
                  [](std::uint8_t value) { return value != 0; })) {
    return Invalid("FlashInfer attention payload reserved bytes are non-zero");
  }

  FlashInferAttentionPayloadView parsed;
  parsed.target_arch = ReadU32(data, 12);
  parsed.variant = static_cast<FlashInferAttentionVariant>(ReadU32(data, 16));
  parsed.num_q_heads = ReadU32(data, 44);
  parsed.num_kv_heads = ReadU32(data, 48);
  parsed.query_length = ReadU32(data, 52);
  parsed.key_length = ReadU32(data, 56);
  parsed.head_dim = ReadU32(data, 60);
  parsed.grid_x = ReadU32(data, 64);
  parsed.block_y = ReadU32(data, 80);
  parsed.shared_bytes = ReadU32(data, 88);
  parsed.query_bytes = ReadU64(data, 96);
  parsed.key_bytes = ReadU64(data, 104);
  parsed.value_bytes = ReadU64(data, 112);
  parsed.mask_bytes = ReadU64(data, 120);
  parsed.output_bytes = ReadU64(data, 128);
  std::copy_n(data + 136, parsed.flashinfer_commit.size(),
              parsed.flashinfer_commit.begin());
  std::copy_n(data + 156, parsed.cccl_commit.size(),
              parsed.cccl_commit.begin());

  const bool common =
      parsed.target_arch == 120 &&
      ReadU32(data, 20) == 2 && ReadU32(data, 24) == 2 &&
      ReadU32(data, 28) == 2 && ReadU32(data, 32) == 4 &&
      ReadU32(data, 40) == 1 &&
      parsed.num_q_heads == 8 && parsed.num_kv_heads == 1 &&
      parsed.head_dim == 256 &&
      ReadU32(data, 68) == 1 && ReadU32(data, 72) == 1 &&
      ReadU32(data, 76) == 32 && parsed.block_y == 4 &&
      ReadU32(data, 84) == 1 && parsed.shared_bytes == 49152 &&
      ReadU32(data, 92) == 0x3d800000U &&
      parsed.flashinfer_commit == kFlashInferCommit &&
      parsed.cccl_commit == kCcclCommit;
  const bool denoise =
      parsed.variant ==
          FlashInferAttentionVariant::kBf16GqaDenoiseDenseBoolToBshd &&
      ReadU32(data, 36) == 1 &&
      parsed.query_length == 50 && parsed.key_length == 1018 &&
      parsed.grid_x == 7 && parsed.query_bytes == 204800 &&
      parsed.key_bytes == 521216 && parsed.value_bytes == 521216 &&
      parsed.mask_bytes == 50900 && parsed.output_bytes == 204800;
  const bool prefix =
      parsed.variant == FlashInferAttentionVariant::kBf16GqaPrefixPadBoolToBshd &&
      ReadU32(data, 36) == 2 &&
      parsed.query_length == 968 && parsed.key_length == 968 &&
      parsed.grid_x == 121 && parsed.query_bytes == 3964928 &&
      parsed.key_bytes == 495616 && parsed.value_bytes == 495616 &&
      parsed.mask_bytes == 968 && parsed.output_bytes == 3964928;
  const bool exact = common && (denoise || prefix);
  if (!exact) {
    return Invalid("FlashInfer attention payload is outside the delivered exact variant");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
