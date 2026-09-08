#pragma once
#include "status.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {
struct ProjectionSplitPayloadView {
  bool rope_pack = false;
  std::uint32_t rows = 0;
  std::array<std::uint32_t, 3> widths{};
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};
Status ParseProjectionSplitPayload(const std::uint8_t*, std::size_t, ProjectionSplitPayloadView*);
}  // namespace aginfer::internal
