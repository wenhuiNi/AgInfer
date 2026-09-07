#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kVisionAttentionPayloadSize = 192;
inline constexpr std::uint16_t kVisionAttentionSchemaMajor = 1;
inline constexpr std::uint16_t kVisionAttentionSchemaMinor = 0;
inline constexpr const char* kVisionAttentionSymbol =
    "aginfer_vision_attention_f32_bshd";

struct VisionAttentionPayloadView {
  std::uint32_t target_arch = 0;
  std::uint32_t query_length = 0;
  std::uint32_t key_length = 0;
  std::uint32_t num_heads = 0;
  std::uint32_t head_dim = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t shared_bytes = 0;
  float scale = 0.0F;
  std::uint64_t query_bytes = 0;
  std::uint64_t key_bytes = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t mask_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseVisionAttentionPayload(const std::uint8_t* data, std::size_t size,
                                   VisionAttentionPayloadView* output);

}  // namespace aginfer::internal
