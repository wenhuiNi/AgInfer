#pragma once
#include "status.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {
struct GeluMulPayloadView {
  std::uint64_t numel = 0, module_bytes = 0;
  std::uint32_t packed_width = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};
Status ParseGeluMulPayload(const std::uint8_t*, std::size_t, GeluMulPayloadView*);
}  // namespace aginfer::internal
