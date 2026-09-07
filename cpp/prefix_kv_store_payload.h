#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kPrefixKvStorePayloadSize = 192;
inline constexpr std::uint16_t kPrefixKvStoreSchemaMajor = 1;
inline constexpr std::uint16_t kPrefixKvStoreSchemaMinor = 0;

struct PrefixKvStorePayloadView {
  std::uint32_t target_arch = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t alignment = 0;
  std::uint64_t key_bytes = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t state_key_bytes = 0;
  std::uint64_t state_value_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
};

Status ParsePrefixKvStorePayload(const std::uint8_t* data, std::size_t size,
                                 PrefixKvStorePayloadView* output);
const char* PrefixKvStoreSymbol() noexcept;

}  // namespace aginfer::internal
