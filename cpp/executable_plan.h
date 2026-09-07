#pragma once

#include "command_stream.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kExecutablePlanHeaderSize = 320;
inline constexpr std::size_t kExecutableValueRecordSize = 160;
inline constexpr std::size_t kExecutablePortRecordSize = 32;
inline constexpr std::size_t kExecutableStateRecordSize = 32;
inline constexpr std::size_t kExecutableProviderRecordSize = 64;
inline constexpr std::uint16_t kExecutablePlanSchemaMajor = 2;
inline constexpr std::uint16_t kExecutablePlanSchemaMinor = 0;
inline constexpr std::uint32_t kExecutableMaxTensorRank = 8;

enum class ExecutableValueRegion : std::uint32_t {
  kUnused = 0,
  kExternalInput = 1,
  kExternalOutput = 2,
  kWeights = 3,
  kState = 4,
  kArena = 5,
  kAlias = 6,
};

enum class ExecutableDType : std::uint32_t {
  kFp32 = 1,
  kFp16 = 2,
  kBf16 = 3,
  kInt32 = 4,
  kBool = 5,
};

enum class ExecutablePortKind : std::uint32_t {
  kInput = 1,
  kOutput = 2,
};

enum class ExecutableStateInit : std::uint32_t { kZero = 1 };

struct ExecutableValueView {
  std::uint32_t value_id = 0;
  ExecutableValueRegion region = ExecutableValueRegion::kUnused;
  ExecutableDType dtype = ExecutableDType::kFp32;
  std::uint32_t rank = 0;
  std::uint32_t alias_of = UINT32_MAX;
  std::uint32_t producer = UINT32_MAX;
  std::uint64_t offset = 0;
  std::uint64_t byte_size = 0;
  std::uint64_t allocation_bytes = 0;
  std::array<std::int64_t, kExecutableMaxTensorRank> shape{};
  std::array<std::uint8_t, 32> sha256{};
};

struct ExecutablePortView {
  std::uint32_t port_id = 0;
  ExecutablePortKind kind = ExecutablePortKind::kInput;
  std::uint32_t value_id = 0;
};

struct ExecutableStateView {
  std::uint32_t state_id = 0;
  std::uint32_t value_id = 0;
  ExecutableStateInit init = ExecutableStateInit::kZero;
};

struct ExecutableProviderView {
  std::uint32_t provider_id = 0;
  std::uint32_t abi_major = 0;
  std::uint32_t abi_minor = 0;
  std::uint32_t tag_mask = 0;
  std::uint32_t command_count = 0;
  std::uint32_t capture_safe_count = 0;
  std::array<std::uint8_t, 32> usage_sha256{};
};

// Non-owning checked view. The caller keeps [data, data + size) immutable.
struct ParsedExecutablePlan {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  std::uint32_t target_arch = 0;
  std::uint32_t value_count = 0;
  std::uint32_t port_count = 0;
  std::uint32_t state_count = 0;
  std::uint32_t provider_count = 0;
  std::uint32_t alignment = 0;
  std::uint64_t arena_bytes = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t weights_bytes = 0;
  std::uint64_t values_offset = 0;
  std::uint64_t ports_offset = 0;
  std::uint64_t states_offset = 0;
  std::uint64_t providers_offset = 0;
  std::uint64_t commands_offset = 0;
  std::array<std::uint8_t, 32> schedule_digest{};
  std::array<std::uint8_t, 32> memory_plan_digest{};
  ParsedCommandStream command_stream;

  bool ReadValue(std::uint32_t index, ExecutableValueView* output) const noexcept;
  bool ReadPort(std::uint32_t index, ExecutablePortView* output) const noexcept;
  bool ReadState(std::uint32_t index, ExecutableStateView* output) const noexcept;
  bool ReadProvider(std::uint32_t index,
                    ExecutableProviderView* output) const noexcept;
  bool RootValue(std::uint32_t value_id, std::uint32_t* output) const noexcept;
};

Status ParseExecutablePlan(const std::uint8_t* data, std::size_t size,
                           std::uint32_t expected_arch,
                           std::uint64_t expected_weight_bytes,
                           ParsedExecutablePlan* output);

}  // namespace aginfer::internal
