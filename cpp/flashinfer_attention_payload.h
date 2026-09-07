#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kFlashInferAttentionPayloadSize = 192;
inline constexpr std::uint16_t kFlashInferAttentionSchemaMajor = 1;
inline constexpr std::uint16_t kFlashInferAttentionSchemaMinor = 0;

enum class FlashInferAttentionVariant : std::uint32_t {
  kBf16GqaDenoiseDenseBoolToBshd = 1,
  kBf16GqaPrefixPadBoolToBshd = 2,
};

struct FlashInferAttentionPayloadView {
  std::uint32_t target_arch = 0;
  FlashInferAttentionVariant variant =
      FlashInferAttentionVariant::kBf16GqaDenoiseDenseBoolToBshd;
  std::uint32_t num_q_heads = 0;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t query_length = 0;
  std::uint32_t key_length = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t grid_x = 0;
  std::uint32_t block_y = 0;
  std::uint32_t shared_bytes = 0;
  std::uint64_t query_bytes = 0;
  std::uint64_t key_bytes = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t mask_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::array<std::uint8_t, 20> flashinfer_commit{};
  std::array<std::uint8_t, 20> cccl_commit{};
};

Status ParseFlashInferAttentionPayload(const std::uint8_t* data,
                                       std::size_t size,
                                       FlashInferAttentionPayloadView* output);

}  // namespace aginfer::internal
