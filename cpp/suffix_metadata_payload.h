#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kSuffixMetadataPayloadSize = 192;

struct SuffixMetadataPayloadView {
  std::uint32_t target_arch = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t pad_alignment = 0;
  std::uint32_t mask_alignment = 0;
  std::uint32_t position_alignment = 0;
  std::uint64_t pad_bytes = 0;
  std::uint64_t mask_bytes = 0;
  std::uint64_t position_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParseSuffixMetadataPayload(const std::uint8_t* data, std::size_t size,
                                  SuffixMetadataPayloadView* output);
const char* SuffixMetadataSymbol() noexcept;

}  // namespace aginfer::internal
