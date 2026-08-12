#pragma once

#include "aginfer/runtime.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace aginfer::internal {

inline constexpr std::uint32_t kPlanSchemaMajor = 1;
inline constexpr std::uint32_t kMaxTensorRank = 8;

enum class PlanDType : std::uint32_t { kFp32 = 1, kFp16 = 2, kBf16 = 3, kFp8E4M3 = 4, kInt32 = 5 };
enum class PlanLocation : std::uint32_t { kHost = 1, kDevice = 2 };
enum class PlanIoKind : std::uint32_t { kInput = 1, kOutput = 2 };
enum class PlanArgKind : std::uint32_t {
  kTensor = 1,
  kScalarU32 = 2,
  kScalarF32 = 3,
  kArenaOffset = 4,
  kWeightsOffset = 5,
};

#pragma pack(push, 1)
struct PlanHeader {
  char magic[8];
  std::uint16_t schema_major;
  std::uint16_t schema_minor;
  std::uint32_t header_size;
  std::uint32_t arch;
  std::uint32_t profile_count;
  std::uint32_t tensor_count;
  std::uint32_t launch_count;
  std::uint32_t argument_count;
  std::uint32_t string_bytes;
  std::uint64_t arena_bytes;
  std::uint64_t workspace_bytes;
  std::uint64_t profiles_offset;
  std::uint64_t tensors_offset;
  std::uint64_t launches_offset;
  std::uint64_t arguments_offset;
  std::uint64_t strings_offset;
  std::uint64_t file_size;
  std::uint8_t reserved[24];
};

struct PlanProfile {
  std::uint32_t name_offset;
  std::uint32_t first_tensor;
  std::uint32_t tensor_count;
  std::uint32_t first_launch;
  std::uint32_t launch_count;
  std::uint32_t flags;
  std::uint8_t reserved[40];
};

struct PlanTensor {
  std::uint32_t name_offset;
  std::uint32_t dtype;
  std::uint32_t location;
  std::uint32_t io_kind;
  std::uint32_t rank;
  std::uint32_t flags;
  std::uint64_t byte_size;
  std::int64_t shape[kMaxTensorRank];
  std::int64_t stride[kMaxTensorRank];
};

struct PlanLaunch {
  std::uint32_t kernel_name_offset;
  std::uint32_t profile_index;
  std::uint32_t first_argument;
  std::uint32_t argument_count;
  std::uint32_t grid[3];
  std::uint32_t block[3];
  std::uint32_t shared_bytes;
  std::uint32_t flags;
  std::uint8_t reserved[16];
};

struct PlanArgument {
  std::uint32_t kind;
  std::uint32_t index;
  std::uint64_t offset;
  std::uint64_t value;
  std::uint64_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(PlanHeader) == 128);
static_assert(sizeof(PlanProfile) == 64);
static_assert(sizeof(PlanTensor) == 160);
static_assert(sizeof(PlanLaunch) == 64);
static_assert(sizeof(PlanArgument) == 32);

struct ParsedPlan {
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  const PlanHeader* header = nullptr;
  const PlanProfile* profiles = nullptr;
  const PlanTensor* tensors = nullptr;
  const PlanLaunch* launches = nullptr;
  const PlanArgument* arguments = nullptr;
  const char* strings = nullptr;

  std::string_view String(std::uint32_t offset) const;
  const PlanProfile* FindProfile(std::string_view name) const;
  std::uint32_t ProfileIndex(const PlanProfile* profile) const;
};

Status ParsePlan(const std::uint8_t* data, std::size_t size,
                 std::uint32_t expected_arch, std::uint64_t weight_bytes,
                 ParsedPlan* output);

}  // namespace aginfer::internal

