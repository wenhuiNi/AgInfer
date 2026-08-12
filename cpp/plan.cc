#include "plan.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<char, 8> kPlanMagic{'A', 'I', 'M', 'P', 'L', 'N', '1', '\0'};

Status Error(const std::string& message) {
  return Status(StatusCode::kCorruptPackage, message);
}

bool AllZero(const std::uint8_t* data, std::size_t size) {
  return std::all_of(data, data + size, [](std::uint8_t value) { return value == 0; });
}

bool RegionValid(std::size_t total, std::uint64_t offset,
                 std::uint64_t count, std::size_t item_size) {
  if (offset > total || count > std::numeric_limits<std::size_t>::max() / item_size) return false;
  const std::size_t bytes = static_cast<std::size_t>(count) * item_size;
  return bytes <= total - static_cast<std::size_t>(offset);
}

bool RangeValid(std::uint32_t total, std::uint32_t first, std::uint32_t count) {
  return first <= total && count <= total - first;
}

std::uint64_t DTypeSize(std::uint32_t dtype) {
  switch (static_cast<PlanDType>(dtype)) {
    case PlanDType::kFp32:
    case PlanDType::kInt32: return 4;
    case PlanDType::kFp16:
    case PlanDType::kBf16: return 2;
    case PlanDType::kFp8E4M3: return 1;
  }
  return 0;
}

Status ValidateTensor(const ParsedPlan& plan, const PlanTensor& tensor,
                      std::uint32_t index) {
  const auto name = plan.String(tensor.name_offset);
  if (name.empty()) return Error("execution plan tensor " + std::to_string(index) + " has an invalid name");
  const std::uint64_t item_size = DTypeSize(tensor.dtype);
  if (item_size == 0) return Error("execution plan tensor " + std::to_string(index) + " has an invalid dtype");
  if (tensor.location != static_cast<std::uint32_t>(PlanLocation::kHost) &&
      tensor.location != static_cast<std::uint32_t>(PlanLocation::kDevice))
    return Error("execution plan tensor " + std::to_string(index) + " has an invalid location");
  if (tensor.io_kind != static_cast<std::uint32_t>(PlanIoKind::kInput) &&
      tensor.io_kind != static_cast<std::uint32_t>(PlanIoKind::kOutput))
    return Error("execution plan tensor " + std::to_string(index) + " has an invalid I/O kind");
  if (tensor.rank == 0 || tensor.rank > kMaxTensorRank || tensor.flags != 0)
    return Error("execution plan tensor " + std::to_string(index) + " has an invalid rank or flags");
  std::uint64_t last_element = 0;
  for (std::uint32_t dimension = 0; dimension < tensor.rank; ++dimension) {
    if (tensor.shape[dimension] <= 0 || tensor.stride[dimension] <= 0)
      return Error("execution plan tensor " + std::to_string(index) + " has a non-positive shape or stride");
    const auto extent = static_cast<std::uint64_t>(tensor.shape[dimension] - 1);
    const auto stride = static_cast<std::uint64_t>(tensor.stride[dimension]);
    if (extent != 0 && stride > (std::numeric_limits<std::uint64_t>::max() - last_element) / extent)
      return Error("execution plan tensor size overflows");
    last_element += extent * stride;
  }
  for (std::uint32_t dimension = tensor.rank; dimension < kMaxTensorRank; ++dimension) {
    if (tensor.shape[dimension] != 0 || tensor.stride[dimension] != 0)
      return Error("execution plan tensor has non-zero dimensions beyond its rank");
  }
  if (last_element == std::numeric_limits<std::uint64_t>::max() ||
      last_element + 1 > std::numeric_limits<std::uint64_t>::max() / item_size ||
      tensor.byte_size < (last_element + 1) * item_size)
    return Error("execution plan tensor byte size is too small");
  return Status::Ok();
}

}  // namespace

std::string_view ParsedPlan::String(std::uint32_t offset) const {
  if (header == nullptr || offset >= header->string_bytes) return {};
  const char* begin = strings + offset;
  const char* end = strings + header->string_bytes;
  const char* terminator = std::find(begin, end, '\0');
  if (terminator == end) return {};
  return std::string_view(begin, static_cast<std::size_t>(terminator - begin));
}

const PlanProfile* ParsedPlan::FindProfile(std::string_view name) const {
  if (header == nullptr) return nullptr;
  for (std::uint32_t index = 0; index < header->profile_count; ++index) {
    if (String(profiles[index].name_offset) == name) return &profiles[index];
  }
  return nullptr;
}

std::uint32_t ParsedPlan::ProfileIndex(const PlanProfile* profile) const {
  return static_cast<std::uint32_t>(profile - profiles);
}

Status ParsePlan(const std::uint8_t* data, std::size_t size,
                 std::uint32_t expected_arch, std::uint64_t weight_bytes,
                 ParsedPlan* output) {
  if (data == nullptr || output == nullptr || size < sizeof(PlanHeader))
    return Error("execution plan is smaller than its fixed header");
  const auto* header = reinterpret_cast<const PlanHeader*>(data);
  if (!std::equal(kPlanMagic.begin(), kPlanMagic.end(), header->magic))
    return Error("bad AIM execution plan magic");
  if (header->schema_major != kPlanSchemaMajor || header->header_size != sizeof(PlanHeader))
    return Error("unsupported AIM execution plan schema");
  if (header->arch != expected_arch)
    return Error("execution plan architecture does not match its AIM variant");
  if (header->file_size != size || !AllZero(header->reserved, sizeof(header->reserved)))
    return Error("invalid execution plan size or reserved bytes");
  if (header->profile_count == 0 || header->tensor_count == 0 || header->launch_count == 0)
    return Error("execution plan must contain profiles, tensors, and launches");
  if (!RegionValid(size, header->profiles_offset, header->profile_count, sizeof(PlanProfile)) ||
      !RegionValid(size, header->tensors_offset, header->tensor_count, sizeof(PlanTensor)) ||
      !RegionValid(size, header->launches_offset, header->launch_count, sizeof(PlanLaunch)) ||
      !RegionValid(size, header->arguments_offset, header->argument_count, sizeof(PlanArgument)) ||
      !RegionValid(size, header->strings_offset, header->string_bytes, 1) || header->string_bytes == 0 ||
      header->strings_offset + header->string_bytes != size)
    return Error("execution plan contains an invalid table region");
  const std::uint64_t expected_tensors = header->profiles_offset +
      static_cast<std::uint64_t>(header->profile_count) * sizeof(PlanProfile);
  const std::uint64_t expected_launches = header->tensors_offset +
      static_cast<std::uint64_t>(header->tensor_count) * sizeof(PlanTensor);
  const std::uint64_t expected_arguments = header->launches_offset +
      static_cast<std::uint64_t>(header->launch_count) * sizeof(PlanLaunch);
  const std::uint64_t expected_strings = header->arguments_offset +
      static_cast<std::uint64_t>(header->argument_count) * sizeof(PlanArgument);
  if (header->profiles_offset != sizeof(PlanHeader) || header->tensors_offset != expected_tensors ||
      header->launches_offset != expected_launches || header->arguments_offset != expected_arguments ||
      header->strings_offset != expected_strings)
    return Error("execution plan tables are not in canonical order");

  ParsedPlan parsed;
  parsed.data = data;
  parsed.size = size;
  parsed.header = header;
  parsed.profiles = reinterpret_cast<const PlanProfile*>(data + header->profiles_offset);
  parsed.tensors = reinterpret_cast<const PlanTensor*>(data + header->tensors_offset);
  parsed.launches = reinterpret_cast<const PlanLaunch*>(data + header->launches_offset);
  parsed.arguments = reinterpret_cast<const PlanArgument*>(data + header->arguments_offset);
  parsed.strings = reinterpret_cast<const char*>(data + header->strings_offset);
  if (parsed.strings[0] != '\0') return Error("execution plan string table has an invalid sentinel");

  std::uint32_t next_tensor = 0;
  std::uint32_t next_launch = 0;
  for (std::uint32_t profile_index = 0; profile_index < header->profile_count; ++profile_index) {
    const PlanProfile& profile = parsed.profiles[profile_index];
    if (parsed.String(profile.name_offset).empty() || profile.first_tensor != next_tensor ||
        profile.first_launch != next_launch || profile.tensor_count == 0 || profile.launch_count == 0 ||
        !RangeValid(header->tensor_count, profile.first_tensor, profile.tensor_count) ||
        !RangeValid(header->launch_count, profile.first_launch, profile.launch_count) ||
        (profile.flags & ~1U) != 0 || !AllZero(profile.reserved, sizeof(profile.reserved)))
      return Error("execution plan contains an invalid profile record");
    for (std::uint32_t previous = 0; previous < profile_index; ++previous) {
      if (parsed.String(parsed.profiles[previous].name_offset) == parsed.String(profile.name_offset))
        return Error("execution plan contains duplicate profile names");
    }
    next_tensor += profile.tensor_count;
    next_launch += profile.launch_count;
  }
  if (next_tensor != header->tensor_count || next_launch != header->launch_count)
    return Error("execution plan profile ranges do not cover their tables");

  for (std::uint32_t index = 0; index < header->tensor_count; ++index) {
    const Status status = ValidateTensor(parsed, parsed.tensors[index], index);
    if (!status.ok()) return status;
  }
  for (std::uint32_t profile_index = 0; profile_index < header->profile_count; ++profile_index) {
    const PlanProfile& profile = parsed.profiles[profile_index];
    std::uint32_t input_count = 0;
    std::uint32_t output_count = 0;
    for (std::uint32_t index = 0; index < profile.tensor_count; ++index) {
      const PlanTensor& tensor = parsed.tensors[profile.first_tensor + index];
      if (tensor.io_kind == static_cast<std::uint32_t>(PlanIoKind::kInput)) ++input_count;
      else ++output_count;
      for (std::uint32_t previous = 0; previous < index; ++previous) {
        const PlanTensor& other = parsed.tensors[profile.first_tensor + previous];
        if (parsed.String(other.name_offset) == parsed.String(tensor.name_offset))
          return Error("execution plan profile contains duplicate tensor names");
      }
    }
    if (input_count == 0 || output_count == 0)
      return Error("execution plan profile must contain inputs and outputs");
  }

  std::uint32_t next_argument = 0;
  for (std::uint32_t index = 0; index < header->launch_count; ++index) {
    const PlanLaunch& launch = parsed.launches[index];
    if (launch.profile_index >= header->profile_count || parsed.String(launch.kernel_name_offset).empty() ||
        launch.first_argument != next_argument ||
        !RangeValid(header->argument_count, launch.first_argument, launch.argument_count) ||
        launch.grid[0] == 0 || launch.grid[1] == 0 || launch.grid[2] == 0 ||
        launch.block[0] == 0 || launch.block[1] == 0 || launch.block[2] == 0 ||
        static_cast<std::uint64_t>(launch.block[0]) * launch.block[1] * launch.block[2] > 1024 ||
        launch.flags != 0 || !AllZero(launch.reserved, sizeof(launch.reserved)))
      return Error("execution plan contains an invalid launch record");
    const PlanProfile& profile = parsed.profiles[launch.profile_index];
    if (index < profile.first_launch || index >= profile.first_launch + profile.launch_count)
      return Error("execution plan launch belongs to the wrong profile");
    for (std::uint32_t arg_index = 0; arg_index < launch.argument_count; ++arg_index) {
      const PlanArgument& argument = parsed.arguments[launch.first_argument + arg_index];
      if (argument.reserved != 0) return Error("execution plan argument has non-zero reserved data");
      switch (static_cast<PlanArgKind>(argument.kind)) {
        case PlanArgKind::kTensor:
          if (argument.index < profile.first_tensor ||
              argument.index >= profile.first_tensor + profile.tensor_count ||
              parsed.tensors[argument.index].location != static_cast<std::uint32_t>(PlanLocation::kDevice))
            return Error("execution plan launch references an invalid device tensor");
          if (argument.offset != 0 || argument.value != 0)
            return Error("execution plan tensor argument has unexpected payload data");
          break;
        case PlanArgKind::kScalarU32:
        case PlanArgKind::kScalarF32:
          if (argument.index != 0 || argument.offset != 0 ||
              argument.value > std::numeric_limits<std::uint32_t>::max())
            return Error("execution plan scalar argument is out of range");
          break;
        case PlanArgKind::kArenaOffset:
          if (argument.index != 0 || argument.value != 0 || argument.offset >= header->arena_bytes)
            return Error("execution plan arena argument is out of bounds");
          break;
        case PlanArgKind::kWeightsOffset:
          if (argument.index != 0 || argument.value != 0 || argument.offset >= weight_bytes)
            return Error("execution plan weight argument is out of bounds");
          break;
        default: return Error("execution plan contains an unknown argument kind");
      }
    }
    next_argument += launch.argument_count;
  }
  if (next_argument != header->argument_count)
    return Error("execution plan launch ranges do not cover the argument table");
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
