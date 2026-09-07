#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kRmsNormPayloadSize = 192;
inline constexpr std::uint16_t kRmsNormSchemaMajor = 1;
inline constexpr std::uint16_t kRmsNormSchemaMinor = 0;

enum class RmsNormVariant : std::uint32_t {
  kF32Rows50Width1024 = 1,
  kBf16Rows968Width2048 = 2,
};

struct RmsNormPayloadView {
  std::uint32_t target_arch = 0;
  RmsNormVariant variant = RmsNormVariant::kF32Rows50Width1024;
  std::uint32_t rows = 0;
  std::uint32_t width = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t shared_bytes = 0;
  std::uint32_t input_alignment = 0;
  std::uint32_t weight_alignment = 0;
  std::uint32_t output_alignment = 0;
  float epsilon = 0.0F;
  std::uint64_t input_bytes = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseRmsNormPayload(const std::uint8_t* data, std::size_t size,
                           RmsNormPayloadView* output);
const char* RmsNormSymbol(RmsNormVariant variant) noexcept;

}  // namespace aginfer::internal
