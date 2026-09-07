#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kPrefixInputPayloadSize = 192;

struct PrefixInputPayloadView {
  std::uint32_t target_arch = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t image_alignment = 0;
  std::uint32_t embedding_alignment = 0;
  std::uint32_t token_alignment = 0;
  std::uint32_t mask_alignment = 0;
  std::uint32_t prefix_alignment = 0;
  std::uint32_t position_alignment = 0;
  std::uint64_t image_bytes = 0;
  std::uint64_t embedding_bytes = 0;
  std::uint64_t token_bytes = 0;
  std::uint64_t prefix_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParsePrefixInputPayload(const std::uint8_t* data, std::size_t size,
                               PrefixInputPayloadView* output);
const char* PrefixInputSymbol() noexcept;

}  // namespace aginfer::internal
