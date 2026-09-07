#include "command_stream.h"

#include "sha256.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'C', 'M', 'D', '1', 0, 0};

Status Error(const std::string& message) {
  return Status(StatusCode::kCorruptPackage, message);
}

std::uint16_t ReadU16(const std::uint8_t* data) noexcept {
  return static_cast<std::uint16_t>(data[0]) |
         static_cast<std::uint16_t>(data[1]) << 8;
}

std::uint32_t ReadU32(const std::uint8_t* data) noexcept {
  return static_cast<std::uint32_t>(data[0]) |
         static_cast<std::uint32_t>(data[1]) << 8 |
         static_cast<std::uint32_t>(data[2]) << 16 |
         static_cast<std::uint32_t>(data[3]) << 24;
}

std::uint64_t ReadU64(const std::uint8_t* data) noexcept {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(data[index]) << (index * 8);
  }
  return value;
}

bool AllZero(const std::uint8_t* data, std::size_t size) noexcept {
  return std::all_of(data, data + size,
                     [](std::uint8_t value) { return value == 0; });
}

bool CheckedMultiply(std::uint64_t left, std::uint64_t right,
                     std::uint64_t* output) noexcept {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
    return false;
  }
  *output = left * right;
  return true;
}

bool CheckedAdd(std::uint64_t left, std::uint64_t right,
                std::uint64_t* output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
  *output = left + right;
  return true;
}

bool KnownArch(std::uint32_t arch) noexcept {
  return arch == 89 || arch == 110 || arch == 120;
}

bool KnownTag(std::uint32_t tag) noexcept {
  return tag >= static_cast<std::uint32_t>(CommandTag::kCudaKernel) &&
         tag <= static_cast<std::uint32_t>(CommandTag::kMemoryCopy);
}

bool KnownAccess(std::uint32_t access) noexcept {
  return access >= static_cast<std::uint32_t>(CommandOperandAccess::kRead) &&
         access <= static_cast<std::uint32_t>(CommandOperandAccess::kReadWrite);
}

}  // namespace

bool ParsedCommandStream::ReadCommand(std::uint32_t index,
                                      CommandRecordView* output) const noexcept {
  if (data == nullptr || output == nullptr || index >= command_count) return false;
  const std::uint8_t* record =
      data + static_cast<std::size_t>(commands_offset) +
      static_cast<std::size_t>(index) * kCommandRecordSize;
  output->tag = static_cast<CommandTag>(ReadU32(record));
  output->provider_id = ReadU32(record + 4);
  output->abi_major = ReadU32(record + 8);
  output->abi_minor = ReadU32(record + 12);
  output->capture_safe = (ReadU32(record + 16) & 1U) != 0;
  output->first_operand = ReadU32(record + 20);
  output->operand_count = ReadU32(record + 24);
  output->payload_size = ReadU32(record + 28);
  output->payload_offset = ReadU64(record + 32);
  output->workspace_offset = ReadU64(record + 40);
  output->workspace_bytes = ReadU64(record + 48);
  std::copy_n(record + 56, output->capability_digest.size(),
              output->capability_digest.begin());
  return true;
}

bool ParsedCommandStream::ReadOperand(std::uint32_t index,
                                      CommandOperandView* output) const noexcept {
  if (data == nullptr || output == nullptr || index >= operand_count) return false;
  const std::uint8_t* record =
      data + static_cast<std::size_t>(operands_offset) +
      static_cast<std::size_t>(index) * kCommandOperandSize;
  output->value_id = ReadU32(record);
  output->access = static_cast<CommandOperandAccess>(ReadU32(record + 4));
  output->byte_offset = ReadU64(record + 8);
  return true;
}

bool ParsedCommandStream::Payload(
    std::uint32_t command_index,
    std::span<const std::uint8_t>* output) const noexcept {
  if (output == nullptr) return false;
  CommandRecordView command;
  if (!ReadCommand(command_index, &command)) return false;
  *output = std::span<const std::uint8_t>(
      data + static_cast<std::size_t>(payload_offset + command.payload_offset),
      command.payload_size);
  return true;
}

Status ParseCommandStream(const std::uint8_t* data, std::size_t size,
                          ParsedCommandStream* output) {
  if (data == nullptr || output == nullptr || size < kCommandStreamHeaderSize) {
    return Error("command stream is smaller than its fixed header");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad command stream magic");
  }
  const std::uint16_t schema_major = ReadU16(data + 8);
  const std::uint16_t schema_minor = ReadU16(data + 10);
  const std::uint32_t header_size = ReadU32(data + 12);
  const std::uint32_t target_arch = ReadU32(data + 16);
  const std::uint32_t command_count = ReadU32(data + 20);
  const std::uint32_t operand_count = ReadU32(data + 24);
  const std::uint32_t flags = ReadU32(data + 28);
  const std::uint32_t reserved_u32 = ReadU32(data + 32);
  const std::uint64_t value_count = ReadU64(data + 36);
  const std::uint64_t arena_bytes = ReadU64(data + 44);
  const std::uint64_t state_bytes = ReadU64(data + 52);
  const std::uint64_t workspace_bytes = ReadU64(data + 60);
  const std::uint64_t commands_offset = ReadU64(data + 68);
  const std::uint64_t operands_offset = ReadU64(data + 76);
  const std::uint64_t payload_offset = ReadU64(data + 84);
  const std::uint64_t payload_bytes = ReadU64(data + 92);
  const std::uint64_t section_size = ReadU64(data + 100);

  if (schema_major != kCommandStreamSchemaMajor ||
      schema_minor > kCommandStreamSchemaMinor) {
    return Error("unsupported command stream schema");
  }
  if (header_size != kCommandStreamHeaderSize || flags != 0 ||
      reserved_u32 != 0 || !AllZero(data + 172, 20)) {
    return Error("invalid command stream header, flags, or reserved bytes");
  }
  if (!KnownArch(target_arch)) return Error("unknown command stream architecture");
  if (command_count == 0 || command_count > kMaxCommands) {
    return Error("command stream command count is outside its bounds");
  }
  if (operand_count == 0 || operand_count > kMaxOperands) {
    return Error("command stream operand count is outside its bounds");
  }
  if (value_count == 0 ||
      value_count > std::numeric_limits<std::uint32_t>::max()) {
    return Error("command stream value count is outside its bounds");
  }

  std::uint64_t command_table_bytes = 0;
  std::uint64_t operand_table_bytes = 0;
  std::uint64_t expected_operands_offset = 0;
  std::uint64_t expected_payload_offset = 0;
  if (!CheckedMultiply(command_count, kCommandRecordSize,
                       &command_table_bytes) ||
      !CheckedMultiply(operand_count, kCommandOperandSize,
                       &operand_table_bytes) ||
      !CheckedAdd(kCommandStreamHeaderSize, command_table_bytes,
                  &expected_operands_offset) ||
      !CheckedAdd(expected_operands_offset, operand_table_bytes,
                  &expected_payload_offset) ||
      commands_offset != kCommandStreamHeaderSize ||
      operands_offset != expected_operands_offset ||
      payload_offset != expected_payload_offset || payload_offset > size ||
      payload_bytes != size - static_cast<std::size_t>(payload_offset) ||
      section_size != size) {
    return Error("command stream tables have invalid or non-canonical bounds");
  }
  if (AllZero(data + 108, 32)) {
    return Error("command stream memory-plan identity is invalid");
  }
  Sha256 hash;
  hash.Update(data + static_cast<std::size_t>(commands_offset),
              size - static_cast<std::size_t>(commands_offset));
  const auto actual_digest = hash.Final();
  if (!std::equal(actual_digest.begin(), actual_digest.end(), data + 140)) {
    return Error("command stream body hash is invalid");
  }

  ParsedCommandStream parsed;
  parsed.data = data;
  parsed.size = size;
  parsed.target_arch = target_arch;
  parsed.command_count = command_count;
  parsed.operand_count = operand_count;
  parsed.value_count = static_cast<std::uint32_t>(value_count);
  parsed.arena_bytes = arena_bytes;
  parsed.state_bytes = state_bytes;
  parsed.workspace_bytes = workspace_bytes;
  parsed.commands_offset = commands_offset;
  parsed.operands_offset = operands_offset;
  parsed.payload_offset = payload_offset;
  parsed.payload_bytes = payload_bytes;
  std::copy_n(data + 108, parsed.memory_plan_digest.size(),
              parsed.memory_plan_digest.begin());

  for (std::uint32_t index = 0; index < operand_count; ++index) {
    CommandOperandView operand;
    parsed.ReadOperand(index, &operand);
    if (!KnownAccess(static_cast<std::uint32_t>(operand.access))) {
      return Error("command stream operand " + std::to_string(index) +
                   " has unknown access");
    }
    if (operand.value_id >= parsed.value_count) {
      return Error("command stream operand " + std::to_string(index) +
                   " has an out-of-range value ID");
    }
  }

  std::uint32_t next_operand = 0;
  std::uint64_t next_payload = 0;
  for (std::uint32_t index = 0; index < command_count; ++index) {
    const std::uint8_t* raw =
        data + static_cast<std::size_t>(commands_offset) +
        static_cast<std::size_t>(index) * kCommandRecordSize;
    const std::uint32_t raw_tag = ReadU32(raw);
    const std::uint32_t command_flags = ReadU32(raw + 16);
    CommandRecordView command;
    parsed.ReadCommand(index, &command);
    if (!KnownTag(raw_tag)) {
      return Error("command stream command " + std::to_string(index) +
                   " has an unknown tag");
    }
    if (command.provider_id == 0 || command.abi_major == 0 ||
        AllZero(command.capability_digest.data(),
                command.capability_digest.size())) {
      return Error("command stream command " + std::to_string(index) +
                   " has an invalid provider receipt");
    }
    if ((command_flags & ~1U) != 0 || !AllZero(raw + 88, 8)) {
      return Error("command stream command " + std::to_string(index) +
                   " has unknown flags or reserved bytes");
    }
    if (command.first_operand != next_operand || command.operand_count == 0 ||
        command.operand_count > operand_count - next_operand) {
      return Error("command stream command " + std::to_string(index) +
                   " has invalid operand bounds");
    }
    if (command.payload_size > kMaxCommandPayload ||
        command.payload_offset != next_payload ||
        command.payload_offset > payload_bytes ||
        command.payload_size > payload_bytes - command.payload_offset) {
      return Error("command stream command " + std::to_string(index) +
                   " has invalid payload bounds");
    }
    if ((command.workspace_bytes == 0 && command.workspace_offset != 0) ||
        command.workspace_offset > workspace_bytes ||
        command.workspace_bytes > workspace_bytes - command.workspace_offset) {
      return Error("command stream command " + std::to_string(index) +
                   " has invalid workspace bounds");
    }
    bool has_write = false;
    for (std::uint32_t operand_index = 0;
         operand_index < command.operand_count; ++operand_index) {
      CommandOperandView operand;
      parsed.ReadOperand(command.first_operand + operand_index, &operand);
      has_write = has_write || operand.access == CommandOperandAccess::kWrite ||
                  operand.access == CommandOperandAccess::kReadWrite;
    }
    if (!has_write) {
      return Error("command stream command " + std::to_string(index) +
                   " has no write operand");
    }
    next_operand += command.operand_count;
    next_payload += command.payload_size;
  }
  if (next_operand != operand_count || next_payload != payload_bytes) {
    return Error("command stream spans do not consume their complete tables");
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
