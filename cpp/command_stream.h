#pragma once

#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aginfer::internal {

inline constexpr std::size_t kCommandStreamHeaderSize = 192;
inline constexpr std::size_t kCommandRecordSize = 96;
inline constexpr std::size_t kCommandOperandSize = 16;
inline constexpr std::uint16_t kCommandStreamSchemaMajor = 1;
inline constexpr std::uint16_t kCommandStreamSchemaMinor = 0;
inline constexpr std::uint32_t kMaxCommands = 1'000'000;
inline constexpr std::uint32_t kMaxOperands = 8'000'000;
inline constexpr std::uint32_t kMaxCommandPayload = 64 * 1024 * 1024;

enum class CommandTag : std::uint32_t {
  kCudaKernel = 1,
  kCublasLtMatmul = 2,
  kAttention = 3,
  kMemoryCopy = 4,
};

enum class CommandOperandAccess : std::uint32_t {
  kRead = 1,
  kWrite = 2,
  kReadWrite = 3,
};

struct CommandRecordView {
  CommandTag tag = CommandTag::kCudaKernel;
  std::uint32_t provider_id = 0;
  std::uint32_t abi_major = 0;
  std::uint32_t abi_minor = 0;
  bool capture_safe = false;
  std::uint32_t first_operand = 0;
  std::uint32_t operand_count = 0;
  std::uint32_t payload_size = 0;
  std::uint64_t payload_offset = 0;
  std::uint64_t workspace_offset = 0;
  std::uint64_t workspace_bytes = 0;
  std::array<std::uint8_t, 32> capability_digest{};
};

struct CommandOperandView {
  std::uint32_t value_id = 0;
  CommandOperandAccess access = CommandOperandAccess::kRead;
  std::uint64_t byte_offset = 0;
};

// This is a non-owning view. The caller must keep [data, data + size) alive and
// unchanged for every ReadCommand, ReadOperand, and Payload call.
struct ParsedCommandStream {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  std::uint32_t target_arch = 0;
  std::uint32_t command_count = 0;
  std::uint32_t operand_count = 0;
  std::uint32_t value_count = 0;
  std::uint64_t arena_bytes = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t commands_offset = 0;
  std::uint64_t operands_offset = 0;
  std::uint64_t payload_offset = 0;
  std::uint64_t payload_bytes = 0;
  std::array<std::uint8_t, 32> memory_plan_digest{};

  bool ReadCommand(std::uint32_t index, CommandRecordView* output) const noexcept;
  bool ReadOperand(std::uint32_t index, CommandOperandView* output) const noexcept;
  bool Payload(std::uint32_t command_index,
               std::span<const std::uint8_t>* output) const noexcept;
};

Status ParseCommandStream(const std::uint8_t* data, std::size_t size,
                          ParsedCommandStream* output);

}  // namespace aginfer::internal
