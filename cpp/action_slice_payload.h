#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kActionSlicePayloadSize = 192;

struct ActionSlicePayloadView {
  std::uint32_t target_arch = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t input_alignment = 0;
  std::uint32_t output_alignment = 0;
  std::uint64_t input_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseActionSlicePayload(const std::uint8_t* data, std::size_t size,
                               ActionSlicePayloadView* output);
const char* ActionSliceSymbol() noexcept;

}  // namespace aginfer::internal
