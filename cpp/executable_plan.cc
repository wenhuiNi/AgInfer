#include "executable_plan.h"

#include "sha256.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'M', 'E', 'X', 'E', '2', 0};
constexpr std::uint32_t kMaxValues = 8'000'000;
constexpr std::uint32_t kMaxPorts = 4096;
constexpr std::uint32_t kMaxStates = 1'000'000;
constexpr std::uint32_t kMaxProviders = 256;

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

std::int64_t ReadI64(const std::uint8_t* data) noexcept {
  return static_cast<std::int64_t>(ReadU64(data));
}

bool AllZero(const std::uint8_t* data, std::size_t size) noexcept {
  return std::all_of(data, data + size,
                     [](std::uint8_t value) { return value == 0; });
}

bool HashEquals(const std::array<std::uint8_t, 32>& actual,
                const std::uint8_t* expected) noexcept {
  unsigned difference = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    difference |= actual[index] ^ expected[index];
  }
  return difference == 0;
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

bool KnownRegion(std::uint32_t region) noexcept { return region <= 6; }

std::uint64_t DTypeBytes(ExecutableDType dtype) noexcept {
  switch (dtype) {
    case ExecutableDType::kFp32:
    case ExecutableDType::kInt32:
      return 4;
    case ExecutableDType::kFp16:
    case ExecutableDType::kBf16:
      return 2;
    case ExecutableDType::kBool:
      return 1;
  }
  return 0;
}

std::uint32_t TagBit(CommandTag tag) noexcept {
  return 1U << static_cast<std::uint32_t>(tag);
}

std::uint32_t KnownTagMask() noexcept {
  return TagBit(CommandTag::kCudaKernel) |
         TagBit(CommandTag::kCublasLtMatmul) |
         TagBit(CommandTag::kAttention) | TagBit(CommandTag::kMemoryCopy);
}

}  // namespace

bool ParsedExecutablePlan::ReadValue(
    std::uint32_t index, ExecutableValueView* output) const noexcept {
  if (data == nullptr || output == nullptr || index >= value_count) return false;
  const std::uint8_t* record =
      data + static_cast<std::size_t>(values_offset) +
      static_cast<std::size_t>(index) * kExecutableValueRecordSize;
  output->value_id = ReadU32(record);
  output->region = static_cast<ExecutableValueRegion>(ReadU32(record + 4));
  output->dtype = static_cast<ExecutableDType>(ReadU32(record + 8));
  output->rank = ReadU32(record + 12);
  output->alias_of = ReadU32(record + 20);
  output->producer = ReadU32(record + 24);
  output->offset = ReadU64(record + 32);
  output->byte_size = ReadU64(record + 40);
  output->allocation_bytes = ReadU64(record + 48);
  for (std::size_t dimension = 0; dimension < output->shape.size();
       ++dimension) {
    output->shape[dimension] = ReadI64(record + 56 + dimension * 8);
  }
  std::copy_n(record + 120, output->sha256.size(), output->sha256.begin());
  return true;
}

bool ParsedExecutablePlan::ReadPort(
    std::uint32_t index, ExecutablePortView* output) const noexcept {
  if (data == nullptr || output == nullptr || index >= port_count) return false;
  const std::uint8_t* record =
      data + static_cast<std::size_t>(ports_offset) +
      static_cast<std::size_t>(index) * kExecutablePortRecordSize;
  output->port_id = ReadU32(record);
  output->kind = static_cast<ExecutablePortKind>(ReadU32(record + 4));
  output->value_id = ReadU32(record + 8);
  return true;
}

bool ParsedExecutablePlan::ReadState(
    std::uint32_t index, ExecutableStateView* output) const noexcept {
  if (data == nullptr || output == nullptr || index >= state_count) return false;
  const std::uint8_t* record =
      data + static_cast<std::size_t>(states_offset) +
      static_cast<std::size_t>(index) * kExecutableStateRecordSize;
  output->state_id = ReadU32(record);
  output->value_id = ReadU32(record + 4);
  output->init = static_cast<ExecutableStateInit>(ReadU32(record + 8));
  return true;
}

bool ParsedExecutablePlan::ReadProvider(
    std::uint32_t index, ExecutableProviderView* output) const noexcept {
  if (data == nullptr || output == nullptr || index >= provider_count) return false;
  const std::uint8_t* record =
      data + static_cast<std::size_t>(providers_offset) +
      static_cast<std::size_t>(index) * kExecutableProviderRecordSize;
  output->provider_id = ReadU32(record);
  output->abi_major = ReadU32(record + 4);
  output->abi_minor = ReadU32(record + 8);
  output->tag_mask = ReadU32(record + 12);
  output->command_count = ReadU32(record + 16);
  output->capture_safe_count = ReadU32(record + 20);
  std::copy_n(record + 32, output->usage_sha256.size(),
              output->usage_sha256.begin());
  return true;
}

bool ParsedExecutablePlan::RootValue(std::uint32_t value_id,
                                     std::uint32_t* output) const noexcept {
  if (output == nullptr || value_id >= value_count) return false;
  std::uint32_t root = value_id;
  for (std::uint32_t depth = 0; depth <= value_count; ++depth) {
    ExecutableValueView value;
    if (!ReadValue(root, &value)) return false;
    if (value.region != ExecutableValueRegion::kAlias) {
      *output = root;
      return true;
    }
    if (value.alias_of >= value_count) return false;
    root = value.alias_of;
  }
  return false;
}

Status ParseExecutablePlan(const std::uint8_t* data, std::size_t size,
                           std::uint32_t expected_arch,
                           std::uint64_t expected_weight_bytes,
                           ParsedExecutablePlan* output) {
  if (data == nullptr || output == nullptr ||
      size < kExecutablePlanHeaderSize) {
    return Error("executable plan is smaller than its fixed header");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data)) {
    return Error("bad executable plan magic");
  }
  const std::uint16_t major = ReadU16(data + 8);
  const std::uint16_t minor = ReadU16(data + 10);
  const std::uint32_t header_size = ReadU32(data + 12);
  const std::uint32_t arch = ReadU32(data + 16);
  const std::uint32_t flags = ReadU32(data + 20);
  const std::uint32_t value_count = ReadU32(data + 24);
  const std::uint32_t port_count = ReadU32(data + 28);
  const std::uint32_t state_count = ReadU32(data + 32);
  const std::uint32_t provider_count = ReadU32(data + 36);
  const std::uint32_t alignment = ReadU32(data + 40);
  const std::uint32_t reserved_u32 = ReadU32(data + 44);
  const std::uint64_t arena_bytes = ReadU64(data + 48);
  const std::uint64_t state_bytes = ReadU64(data + 56);
  const std::uint64_t workspace_bytes = ReadU64(data + 64);
  const std::uint64_t weights_bytes = ReadU64(data + 72);
  const std::uint64_t values_offset = ReadU64(data + 80);
  const std::uint64_t ports_offset = ReadU64(data + 88);
  const std::uint64_t states_offset = ReadU64(data + 96);
  const std::uint64_t providers_offset = ReadU64(data + 104);
  const std::uint64_t commands_offset = ReadU64(data + 112);
  const std::uint64_t command_bytes = ReadU64(data + 120);
  const std::uint64_t section_size = ReadU64(data + 128);

  if (major != kExecutablePlanSchemaMajor ||
      minor > kExecutablePlanSchemaMinor) {
    return Error("unsupported executable plan schema");
  }
  if (header_size != kExecutablePlanHeaderSize || flags != 0 ||
      reserved_u32 != 0 || !AllZero(data + 264, 56)) {
    return Error("invalid executable plan header, flags, or reserved bytes");
  }
  if (!KnownArch(arch) || arch != expected_arch) {
    return Error("executable plan architecture differs from its AIM variant");
  }
  if (weights_bytes == 0 || weights_bytes != expected_weight_bytes) {
    return Error("executable plan weight size differs from its AIM variant");
  }
  if (value_count == 0 || value_count > kMaxValues || port_count == 0 ||
      port_count > kMaxPorts || state_count > kMaxStates ||
      provider_count == 0 || provider_count > kMaxProviders ||
      alignment == 0 || alignment > (1U << 20) ||
      (alignment & (alignment - 1)) != 0) {
    return Error("executable plan counts or alignment are outside their bounds");
  }

  std::uint64_t value_bytes = 0;
  std::uint64_t port_bytes = 0;
  std::uint64_t state_table_bytes = 0;
  std::uint64_t provider_bytes = 0;
  std::uint64_t expected_ports = 0;
  std::uint64_t expected_states = 0;
  std::uint64_t expected_providers = 0;
  std::uint64_t expected_commands = 0;
  std::uint64_t expected_size = 0;
  if (!CheckedMultiply(value_count, kExecutableValueRecordSize,
                       &value_bytes) ||
      !CheckedMultiply(port_count, kExecutablePortRecordSize, &port_bytes) ||
      !CheckedMultiply(state_count, kExecutableStateRecordSize,
                       &state_table_bytes) ||
      !CheckedMultiply(provider_count, kExecutableProviderRecordSize,
                       &provider_bytes) ||
      !CheckedAdd(kExecutablePlanHeaderSize, value_bytes, &expected_ports) ||
      !CheckedAdd(expected_ports, port_bytes, &expected_states) ||
      !CheckedAdd(expected_states, state_table_bytes, &expected_providers) ||
      !CheckedAdd(expected_providers, provider_bytes, &expected_commands) ||
      !CheckedAdd(expected_commands, command_bytes, &expected_size) ||
      values_offset != kExecutablePlanHeaderSize ||
      ports_offset != expected_ports || states_offset != expected_states ||
      providers_offset != expected_providers ||
      commands_offset != expected_commands || section_size != size ||
      expected_size != size) {
    return Error("executable plan tables have invalid or non-canonical bounds");
  }
  if (AllZero(data + 136, 32) || AllZero(data + 168, 32) ||
      AllZero(data + 200, 32) || AllZero(data + 232, 32)) {
    return Error("executable plan has a zero identity digest");
  }
  Sha256 body_hash;
  body_hash.Update(data + kExecutablePlanHeaderSize,
                   size - kExecutablePlanHeaderSize);
  if (!HashEquals(body_hash.Final(), data + 232)) {
    return Error("executable plan body checksum mismatch");
  }
  Sha256 command_hash;
  command_hash.Update(data + static_cast<std::size_t>(commands_offset),
                      static_cast<std::size_t>(command_bytes));
  if (!HashEquals(command_hash.Final(), data + 200)) {
    return Error("executable plan command-stream checksum mismatch");
  }

  ParsedExecutablePlan parsed;
  parsed.data = data;
  parsed.size = size;
  parsed.target_arch = arch;
  parsed.value_count = value_count;
  parsed.port_count = port_count;
  parsed.state_count = state_count;
  parsed.provider_count = provider_count;
  parsed.alignment = alignment;
  parsed.arena_bytes = arena_bytes;
  parsed.state_bytes = state_bytes;
  parsed.workspace_bytes = workspace_bytes;
  parsed.weights_bytes = weights_bytes;
  parsed.values_offset = values_offset;
  parsed.ports_offset = ports_offset;
  parsed.states_offset = states_offset;
  parsed.providers_offset = providers_offset;
  parsed.commands_offset = commands_offset;
  std::copy_n(data + 136, 32, parsed.schedule_digest.begin());
  std::copy_n(data + 168, 32, parsed.memory_plan_digest.begin());
  Status status = ParseCommandStream(
      data + static_cast<std::size_t>(commands_offset),
      static_cast<std::size_t>(command_bytes), &parsed.command_stream);
  if (!status.ok()) return Error("invalid executable plan command stream: " + status.message());
  if (parsed.command_stream.target_arch != arch ||
      parsed.command_stream.value_count != value_count ||
      parsed.command_stream.arena_bytes != arena_bytes ||
      parsed.command_stream.state_bytes != state_bytes ||
      parsed.command_stream.workspace_bytes != workspace_bytes ||
      !std::equal(parsed.memory_plan_digest.begin(),
                  parsed.memory_plan_digest.end(),
                  parsed.command_stream.memory_plan_digest.begin())) {
    return Error("executable plan command stream differs from its header");
  }

  std::vector<std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>>
      weight_intervals;
  for (std::uint32_t index = 0; index < value_count; ++index) {
    const std::uint8_t* raw =
        data + static_cast<std::size_t>(values_offset) +
        static_cast<std::size_t>(index) * kExecutableValueRecordSize;
    ExecutableValueView value;
    parsed.ReadValue(index, &value);
    const std::uint32_t raw_region = ReadU32(raw + 4);
    const std::uint32_t raw_dtype = ReadU32(raw + 8);
    if (value.value_id != index || !KnownRegion(raw_region) ||
        DTypeBytes(value.dtype) == 0 || value.rank == 0 ||
        value.rank > kExecutableMaxTensorRank || ReadU32(raw + 16) != 0 ||
        ReadU32(raw + 28) != 0 || !AllZero(raw + 152, 8)) {
      return Error("executable plan value " + std::to_string(index) +
                   " has an invalid tensor contract");
    }
    (void)raw_dtype;
    std::uint64_t element_count = 1;
    for (std::uint32_t dimension = 0; dimension < value.rank; ++dimension) {
      if (value.shape[dimension] <= 0 ||
          !CheckedMultiply(element_count,
                           static_cast<std::uint64_t>(value.shape[dimension]),
                           &element_count)) {
        return Error("executable plan value tensor size overflows");
      }
    }
    for (std::uint32_t dimension = value.rank;
         dimension < kExecutableMaxTensorRank; ++dimension) {
      if (value.shape[dimension] != 0) {
        return Error("executable plan value has dimensions beyond its rank");
      }
    }
    std::uint64_t required_bytes = 0;
    if (!CheckedMultiply(element_count, DTypeBytes(value.dtype),
                         &required_bytes) ||
        required_bytes != value.byte_size) {
      return Error("executable plan value byte size is invalid");
    }
    const bool has_digest = !AllZero(value.sha256.data(), value.sha256.size());
    if (value.region == ExecutableValueRegion::kAlias) {
      if (value.alias_of >= value_count || value.alias_of == index ||
          value.offset != 0 || value.allocation_bytes != 0 || has_digest) {
        return Error("executable plan alias value owns storage");
      }
    } else {
      if (value.alias_of != UINT32_MAX) {
        return Error("executable plan owned value has an alias target");
      }
      std::uint64_t limit = 0;
      switch (value.region) {
        case ExecutableValueRegion::kUnused:
          if (value.offset != 0 || value.allocation_bytes != 0 || has_digest)
            return Error("executable plan unused value owns storage");
          break;
        case ExecutableValueRegion::kExternalInput:
        case ExecutableValueRegion::kExternalOutput:
          if (value.offset != 0 ||
              value.allocation_bytes != value.byte_size || has_digest)
            return Error("executable plan external value storage is invalid");
          break;
        case ExecutableValueRegion::kWeights:
          limit = weights_bytes;
          break;
        case ExecutableValueRegion::kState:
          limit = state_bytes;
          break;
        case ExecutableValueRegion::kArena:
          limit = arena_bytes;
          break;
        case ExecutableValueRegion::kAlias:
          break;
      }
      if (value.region == ExecutableValueRegion::kWeights ||
          value.region == ExecutableValueRegion::kState ||
          value.region == ExecutableValueRegion::kArena) {
        if (value.allocation_bytes < value.byte_size || value.offset > limit ||
            value.allocation_bytes > limit - value.offset ||
            value.offset % alignment != 0 ||
            (value.region == ExecutableValueRegion::kWeights) != has_digest) {
          return Error("executable plan value exceeds its storage region");
        }
        if (value.region == ExecutableValueRegion::kWeights) {
          weight_intervals.emplace_back(
              value.offset, value.offset + value.allocation_bytes, index);
        }
      }
    }
  }
  std::sort(weight_intervals.begin(), weight_intervals.end());
  for (std::size_t index = 1; index < weight_intervals.size(); ++index) {
    if (std::get<0>(weight_intervals[index]) <
        std::get<1>(weight_intervals[index - 1])) {
      return Error("executable plan packed weight spans overlap");
    }
  }
  for (std::uint32_t index = 0; index < value_count; ++index) {
    std::uint32_t root = 0;
    if (!parsed.RootValue(index, &root)) {
      return Error("executable plan value aliases contain a cycle");
    }
    ExecutableValueView value;
    ExecutableValueView root_value;
    parsed.ReadValue(index, &value);
    parsed.ReadValue(root, &root_value);
    if (value.region == ExecutableValueRegion::kAlias &&
        value.byte_size != root_value.byte_size) {
      return Error("executable plan alias changes tensor byte size");
    }
  }

  bool has_output = false;
  std::vector<bool> port_values(value_count, false);
  for (std::uint32_t index = 0; index < port_count; ++index) {
    const std::uint8_t* raw =
        data + static_cast<std::size_t>(ports_offset) +
        static_cast<std::size_t>(index) * kExecutablePortRecordSize;
    ExecutablePortView port;
    parsed.ReadPort(index, &port);
    std::uint32_t root = 0;
    if (port.port_id != index || port.value_id >= value_count ||
        ReadU32(raw + 12) != 2 || ReadU32(raw + 16) != 0 ||
        ReadU32(raw + 20) != 0 || ReadU32(raw + 24) != 0 ||
        ReadU32(raw + 28) != 0 ||
        (port.kind != ExecutablePortKind::kInput &&
         port.kind != ExecutablePortKind::kOutput) ||
        port_values[port.value_id] ||
        !parsed.RootValue(port.value_id, &root)) {
      return Error("executable plan port " + std::to_string(index) +
                   " has an invalid value contract");
    }
    ExecutableValueView root_value;
    parsed.ReadValue(root, &root_value);
    const ExecutableValueRegion expected =
        port.kind == ExecutablePortKind::kInput
            ? ExecutableValueRegion::kExternalInput
            : ExecutableValueRegion::kExternalOutput;
    if (root_value.region != expected || port_values[root]) {
      return Error("executable plan port storage region is invalid");
    }
    port_values[root] = true;
    has_output = has_output || port.kind == ExecutablePortKind::kOutput;
  }
  if (!has_output) {
    return Error("executable plan requires output ports");
  }

  std::vector<bool> state_values(value_count, false);
  for (std::uint32_t index = 0; index < state_count; ++index) {
    const std::uint8_t* raw =
        data + static_cast<std::size_t>(states_offset) +
        static_cast<std::size_t>(index) * kExecutableStateRecordSize;
    ExecutableStateView state;
    parsed.ReadState(index, &state);
    std::uint32_t root = 0;
    if (state.state_id != index || state.value_id >= value_count ||
        state.init != ExecutableStateInit::kZero || ReadU32(raw + 12) != 0 ||
        ReadU64(raw + 16) != 0 || ReadU64(raw + 24) != 0 ||
        state_values[state.value_id] ||
        !parsed.RootValue(state.value_id, &root)) {
      return Error("executable plan state " + std::to_string(index) +
                   " has an invalid value contract");
    }
    ExecutableValueView root_value;
    parsed.ReadValue(root, &root_value);
    if (root_value.region != ExecutableValueRegion::kState || state_values[root]) {
      return Error("executable plan state storage region is invalid");
    }
    state_values[root] = true;
  }

  for (std::uint32_t index = 0; index < value_count; ++index) {
    ExecutableValueView value; parsed.ReadValue(index, &value);
    if ((value.region == ExecutableValueRegion::kExternalInput ||
         value.region == ExecutableValueRegion::kExternalOutput) && !port_values[index])
      return Error("executable plan does not bind every external storage root");
    if (value.region == ExecutableValueRegion::kState && !state_values[index])
      return Error("executable plan does not initialize every state storage root");
  }
  std::uint32_t previous_provider = 0;
  for (std::uint32_t index = 0; index < provider_count; ++index) {
    const std::uint8_t* raw =
        data + static_cast<std::size_t>(providers_offset) +
        static_cast<std::size_t>(index) * kExecutableProviderRecordSize;
    ExecutableProviderView provider;
    parsed.ReadProvider(index, &provider);
    if (provider.provider_id <= previous_provider || provider.abi_major == 0 ||
        provider.tag_mask == 0 ||
        (provider.tag_mask & ~KnownTagMask()) != 0 ||
        provider.command_count == 0 ||
        provider.capture_safe_count > provider.command_count ||
        ReadU32(raw + 24) != 0 || ReadU32(raw + 28) != 0 ||
        AllZero(provider.usage_sha256.data(), provider.usage_sha256.size())) {
      return Error("executable plan provider descriptor is invalid");
    }
    std::uint32_t actual_count = 0;
    std::uint32_t actual_capture_count = 0;
    std::uint32_t actual_tag_mask = 0;
    Sha256 usage_hash;
    for (std::uint32_t command_index = 0;
         command_index < parsed.command_stream.command_count;
         ++command_index) {
      CommandRecordView command;
      parsed.command_stream.ReadCommand(command_index, &command);
      if (command.provider_id != provider.provider_id) continue;
      if (command.abi_major != provider.abi_major ||
          command.abi_minor != provider.abi_minor) {
        return Error("executable plan provider ABI differs across commands");
      }
      ++actual_count;
      actual_capture_count += command.capture_safe ? 1U : 0U;
      actual_tag_mask |= TagBit(command.tag);
      usage_hash.Update(command.capability_digest.data(),
                        command.capability_digest.size());
      std::span<const std::uint8_t> payload;
      parsed.command_stream.Payload(command_index, &payload);
      Sha256 payload_hash;
      payload_hash.Update(payload.data(), payload.size());
      const auto payload_digest = payload_hash.Final();
      usage_hash.Update(payload_digest.data(), payload_digest.size());
    }
    if (actual_count != provider.command_count ||
        actual_capture_count != provider.capture_safe_count ||
        actual_tag_mask != provider.tag_mask ||
        !HashEquals(usage_hash.Final(), provider.usage_sha256.data())) {
      return Error("executable plan provider table differs from its command stream");
    }
    previous_provider = provider.provider_id;
  }
  for (std::uint32_t command_index = 0;
       command_index < parsed.command_stream.command_count; ++command_index) {
    CommandRecordView command;
    parsed.command_stream.ReadCommand(command_index, &command);
    bool provider_found = false;
    for (std::uint32_t provider_index = 0; provider_index < provider_count;
         ++provider_index) {
      ExecutableProviderView provider;
      parsed.ReadProvider(provider_index, &provider);
      provider_found = provider_found || provider.provider_id == command.provider_id;
    }
    if (!provider_found) {
      return Error("executable plan command names an undeclared provider");
    }
    for (std::uint32_t operand_index = 0;
         operand_index < command.operand_count; ++operand_index) {
      CommandOperandView operand;
      parsed.command_stream.ReadOperand(command.first_operand + operand_index,
                                        &operand);
      std::uint32_t root = 0;
      if (!parsed.RootValue(operand.value_id, &root)) {
        return Error("executable plan command operand has an invalid alias");
      }
      ExecutableValueView value;
      parsed.ReadValue(root, &value);
      if (value.region == ExecutableValueRegion::kUnused ||
          operand.byte_offset >= value.byte_size) {
        return Error("executable plan command operand is out of bounds");
      }
    }
  }
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
