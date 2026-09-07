#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kAdaptiveRmsNormPayloadSize = 192;
inline constexpr std::uint16_t kAdaptiveRmsNormSchemaMajor = 1;
inline constexpr std::uint16_t kAdaptiveRmsNormSchemaMinor = 0;

struct AdaptiveRmsNormPayloadView {
  std::uint32_t target_arch = 0;
  std::uint32_t rows = 0;
  std::uint32_t width = 0;
  std::uint32_t modulation_width = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t shared_bytes = 0;
  std::uint32_t hidden_alignment = 0;
  std::uint32_t modulation_alignment = 0;
  std::uint32_t normalized_alignment = 0;
  std::uint32_t gate_alignment = 0;
  float epsilon = 0.0F;
  std::uint64_t hidden_bytes = 0;
  std::uint64_t modulation_bytes = 0;
  std::uint64_t normalized_bytes = 0;
  std::uint64_t gate_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseAdaptiveRmsNormPayload(const std::uint8_t* data, std::size_t size,
                                   AdaptiveRmsNormPayloadView* output);
const char* AdaptiveRmsNormSymbol() noexcept;

}  // namespace aginfer::internal
