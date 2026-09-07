#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kLayerNormPayloadSize = 192;
inline constexpr std::uint16_t kLayerNormSchemaMajor = 1;
inline constexpr std::uint16_t kLayerNormSchemaMinor = 0;
inline constexpr const char* kLayerNormSymbol =
    "aginfer_layer_norm_f32_1152";

struct LayerNormPayloadView {
  std::uint32_t target_arch = 0;
  std::uint32_t rows = 0;
  std::uint32_t width = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t shared_bytes = 0;
  std::uint32_t input_alignment = 0;
  std::uint32_t parameter_alignment = 0;
  std::uint32_t output_alignment = 0;
  float epsilon = 0.0F;
  std::uint64_t input_bytes = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t bias_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseLayerNormPayload(const std::uint8_t* data, std::size_t size,
                             LayerNormPayloadView* output);

}  // namespace aginfer::internal
