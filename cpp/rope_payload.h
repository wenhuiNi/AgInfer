#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kRopePayloadSize = 192;
inline constexpr std::uint16_t kRopeSchemaMajor = 1;
inline constexpr std::uint16_t kRopeSchemaMinor = 0;

enum class RopeVariant : std::uint32_t {
  kBf16Sequence968Heads8 = 1,
  kBf16Sequence968Heads1 = 2,
  kBf16Sequence50Heads8 = 3,
  kBf16Sequence50Heads1 = 4,
};

struct RopePayloadView {
  std::uint32_t target_arch = 0;
  RopeVariant variant = RopeVariant::kBf16Sequence968Heads8;
  std::uint32_t heads = 0;
  std::uint32_t sequence = 0;
  std::uint32_t head_dim = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t shared_bytes = 0;
  std::uint32_t input_alignment = 0;
  std::uint32_t position_alignment = 0;
  std::uint32_t output_alignment = 0;
  float theta = 0.0F;
  std::uint64_t input_bytes = 0;
  std::uint64_t position_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseRopePayload(const std::uint8_t* data, std::size_t size,
                        RopePayloadView* output);
const char* RopeSymbol(RopeVariant variant) noexcept;

}  // namespace aginfer::internal
