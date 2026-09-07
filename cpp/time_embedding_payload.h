#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kTimeEmbeddingPayloadSize = 192;

struct TimeEmbeddingPayloadView {
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

Status ParseTimeEmbeddingPayload(const std::uint8_t* data, std::size_t size,
                                 TimeEmbeddingPayloadView* output);
const char* TimeEmbeddingSymbol() noexcept;

}  // namespace aginfer::internal
