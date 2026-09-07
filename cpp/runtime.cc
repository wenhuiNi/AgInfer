#include "aginfer/c_api.h"

#include "cuda_driver.h"
#include "executable_session.h"
#include "plan.h"
#include "sha256.h"
#include "status.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using aginfer::internal::Status;
using aginfer::internal::StatusCode;
using aginfer::internal::StatusOr;

#pragma pack(push, 1)
struct Header {
  char magic[8];
  std::uint16_t schema_major;
  std::uint16_t schema_minor;
  std::uint32_t header_size;
  std::uint32_t runtime_abi;
  std::uint32_t platform;
  std::uint32_t endian_tag;
  std::uint32_t variant_count;
  std::uint64_t file_size;
  std::uint64_t manifest_offset;
  std::uint64_t manifest_size;
  std::uint64_t graph_offset;
  std::uint64_t graph_size;
  std::uint64_t tensor_offset;
  std::uint64_t tensor_size;
  std::uint64_t compatibility_offset;
  std::uint64_t compatibility_size;
  std::uint64_t directory_offset;
  std::uint64_t directory_size;
  std::uint8_t directory_sha[32];
  std::uint8_t file_sha[32];
  std::uint8_t manifest_sha[32];
  std::uint8_t graph_sha[32];
  std::uint8_t compatibility_sha[32];
  std::uint8_t reserved[40];
};

struct Variant {
  std::uint32_t arch;
  std::uint32_t flags;
  std::uint64_t kernel_offset;
  std::uint64_t kernel_size;
  std::uint64_t weight_offset;
  std::uint64_t weight_size;
  std::uint64_t plan_offset;
  std::uint64_t plan_size;
  std::uint8_t kernel_sha[32];
  std::uint8_t weight_sha[32];
  std::uint8_t plan_sha[32];
  std::uint8_t reserved[40];
};

struct CompatibilityHeader {
  char magic[8];
  std::uint16_t schema_major;
  std::uint16_t schema_minor;
  std::uint32_t header_size;
  std::uint32_t cuda_driver_min;
  std::uint32_t cuda_driver_max;
  std::uint32_t cuda_runtime_min;
  std::uint32_t cuda_runtime_max;
  std::uint32_t provider_count;
  std::uint32_t flags;
  std::uint64_t providers_offset;
  std::uint64_t section_size;
  std::uint8_t reserved[8];
};

struct ProviderRequirement {
  std::uint32_t provider_id;
  std::uint32_t abi_min;
  std::uint32_t abi_max;
  std::uint32_t flags;
  std::uint8_t reserved[16];
};
#pragma pack(pop)

static_assert(sizeof(Header) == 320);
static_assert(sizeof(Variant) == 192);
static_assert(sizeof(CompatibilityHeader) == 64);
static_assert(sizeof(ProviderRequirement) == 32);
static_assert(offsetof(Header, file_sha) == 152);

constexpr std::array<char, 8> kMagic{'A', 'I', 'M', 'A', 'O', 'T', '2', '\0'};
constexpr std::array<char, 8> kCompatibilityMagic{'A', 'I', 'M', 'C', 'M', 'P', '1', '\0'};
constexpr std::uint32_t kEndianTag = 0x01020304;
constexpr std::uint32_t kMaxProviderRequirements = 256;
#ifdef AGINFER_VERIFY_AIM_CHECKSUMS
constexpr std::size_t kFileHashOffset = offsetof(Header, file_sha);
#endif

struct ProviderAbiValue {
  std::uint32_t provider_id = 0;
  std::uint32_t abi_version = 0;
};

struct BoundTensor {
  bool bound = false;
  void* data = nullptr;
};

thread_local std::string g_last_error;

Status Error(StatusCode code, std::string message) {
  return Status(code, std::move(message));
}

bool AllZero(const std::uint8_t* data, std::size_t size) {
  return std::all_of(data, data + size,
                     [](std::uint8_t value) { return value == 0; });
}

bool RegionValid(std::size_t total, std::uint64_t offset,
                 std::uint64_t size) {
  return offset >= sizeof(Header) && size > 0 && offset <= total &&
         size <= total - offset;
}

bool RangeContains(std::uint32_t value, std::uint32_t minimum,
                   std::uint32_t maximum) {
  return value >= minimum && (maximum == 0 || value <= maximum);
}

bool VersionRangeValid(std::uint32_t minimum, std::uint32_t maximum,
                       bool required) {
  if (required && minimum == 0) return false;
  if (minimum == 0) return maximum == 0;
  return maximum == 0 || maximum >= minimum;
}

#ifdef AGINFER_VERIFY_AIM_CHECKSUMS
std::array<std::uint8_t, 32> Hash(const std::uint8_t* data,
                                  std::size_t size) {
  aginfer::internal::Sha256 hash;
  hash.Update(data, size);
  return hash.Final();
}

bool HashEquals(const std::uint8_t* actual, const std::uint8_t* expected) {
  unsigned difference = 0;
  for (std::size_t index = 0; index < 32; ++index) {
    difference |= actual[index] ^ expected[index];
  }
  return difference == 0;
}

bool CheckHash(const std::uint8_t* data, std::size_t size,
               const std::uint8_t expected[32]) {
  const auto digest = Hash(data, size);
  return HashEquals(digest.data(), expected);
}

std::array<std::uint8_t, 32> FileHash(const std::uint8_t* data,
                                      std::size_t size) {
  aginfer::internal::Sha256 hash;
  hash.Update(data, kFileHashOffset);
  const std::array<std::uint8_t, 32> zeros{};
  hash.Update(zeros.data(), zeros.size());
  hash.Update(data + kFileHashOffset + zeros.size(),
              size - kFileHashOffset - zeros.size());
  return hash.Final();
}

#endif  // AGINFER_VERIFY_AIM_CHECKSUMS

std::uint32_t HostPlatformId() {
#if defined(__linux__) && defined(__x86_64__)
  return AI_PLATFORM_LINUX_X86_64_GNU;
#elif defined(__linux__) && defined(__aarch64__)
  return AI_PLATFORM_LINUX_AARCH64_SBSA;
#else
  return 0;
#endif
}

const char* HostPlatformName() {
#if defined(__linux__) && defined(__x86_64__)
  return "linux-x86_64-gnu";
#elif defined(__linux__) && defined(__aarch64__)
  return "linux-aarch64-sbsa";
#else
  return "unsupported";
#endif
}

bool ArchMatchesPlatform(std::uint32_t platform, std::uint32_t arch) {
  return (platform == AI_PLATFORM_LINUX_X86_64_GNU &&
          (arch == 89 || arch == 120)) ||
         (platform == AI_PLATFORM_LINUX_AARCH64_SBSA && arch == 110);
}

bool ContainsPtx(const std::uint8_t* data, std::size_t size) {
  constexpr std::array<std::string_view, 4> markers{
      ".version ", ".target sm_", ".entry ", ".visible .entry"};
  const std::string_view content(
      reinterpret_cast<const char*>(data),
      std::min<std::size_t>(size, 1024 * 1024));
  return std::any_of(markers.begin(), markers.end(),
                     [&](std::string_view marker) {
                       return content.find(marker) != std::string_view::npos;
                     });
}

std::uint16_t LoadLittle16(const std::uint8_t* data) {
  return static_cast<std::uint16_t>(data[0]) |
         (static_cast<std::uint16_t>(data[1]) << 8);
}

std::uint32_t LoadLittle32(const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) |
         (static_cast<std::uint32_t>(data[1]) << 8) |
         (static_cast<std::uint32_t>(data[2]) << 16) |
         (static_cast<std::uint32_t>(data[3]) << 24);
}

bool IsExactCubin(const std::uint8_t* data, std::size_t size,
                  std::uint32_t arch) {
  constexpr std::array<std::uint8_t, 7> elf64_le{0x7f, 'E', 'L', 'F',
                                                 2, 1, 1};
  if (size < 64 || !std::equal(elf64_le.begin(), elf64_le.end(), data) ||
      LoadLittle16(data + 18) != 190) {
    return false;
  }
  const std::uint32_t flags = LoadLittle32(data + 48);
  return (flags & 0xff) == arch || ((flags >> 8) & 0xff) == arch ||
         ((flags >> 16) & 0xff) == arch;
}

void* OpenLibrary(std::initializer_list<const char*> names) {
  for (const char* name : names) {
    if (void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL)) return handle;
  }
  return nullptr;
}

Status ValidateStruct(const void* pointer, std::size_t minimum_size,
                      const char* label) {
  if (pointer == nullptr) {
    return Error(StatusCode::kInvalidArgument,
                 std::string(label) + " must not be null");
  }
  const auto* fields = static_cast<const std::uint32_t*>(pointer);
  if (fields[0] < minimum_size) {
    return Error(StatusCode::kInvalidArgument,
                 std::string(label) + " has a short struct_size");
  }
  if (fields[1] != AI_STRUCT_VERSION_1) {
    return Error(StatusCode::kIncompatibleAbi,
                 std::string(label) + " has an unsupported struct_version");
  }
  return Status::Ok();
}

ai_status Record(const Status& status) {
  if (status.ok()) return AI_STATUS_OK;
  g_last_error = status.message();
  return static_cast<ai_status>(status.code());
}

ai_status Fail(StatusCode code, const std::string& message) {
  return Record(Error(code, message));
}

template <typename Function>
ai_status Guard(Function&& function) noexcept {
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return Fail(StatusCode::kOutOfMemory, "host allocation failed");
  } catch (const std::exception& error) {
    return Fail(StatusCode::kInvalidState,
                std::string("unexpected runtime exception: ") + error.what());
  } catch (...) {
    return Fail(StatusCode::kInvalidState, "unknown runtime exception");
  }
}

}  // namespace

struct ai_runtime {
  std::uint32_t cuda_arch = 0;
  std::uint32_t cuda_driver_version = 0;
  std::uint32_t cuda_runtime_version = 0;
  std::vector<ProviderAbiValue> provider_abis;
};

struct ai_model {
  int fd = -1;
  std::size_t size = 0;
  const std::uint8_t* data = nullptr;
  const Header* header = nullptr;
  const Variant* variants = nullptr;
  const CompatibilityHeader* compatibility = nullptr;
  const ProviderRequirement* providers = nullptr;

  ~ai_model() {
    if (data != nullptr) munmap(const_cast<std::uint8_t*>(data), size);
    if (fd >= 0) close(fd);
  }
};

struct ai_session {
  ai_runtime* runtime = nullptr;
  ai_model* model = nullptr;
  const Variant* variant = nullptr;
  aginfer::internal::ParsedPlan plan;
  std::unique_ptr<aginfer::internal::ExecutableSession> executable;
  const aginfer::internal::PlanProfile* profile = nullptr;
  std::unique_ptr<aginfer::internal::CudaDriver> cuda;
  aginfer::internal::CuModule module = nullptr;
  aginfer::internal::CuDevicePtr weights = 0;
  aginfer::internal::CuDevicePtr arena = 0;
  aginfer::internal::CuDevicePtr workspace = 0;
  std::vector<aginfer::internal::CuFunction> functions;
  std::vector<BoundTensor> tensor_bindings;
  std::vector<std::uint64_t> argument_values;
  std::vector<void*> kernel_parameters;
  std::string last_error;
  bool prepared = false;

  ~ai_session() {
    if (cuda == nullptr) return;
    cuda->MakeCurrent();
    cuda->Free(workspace);
    cuda->Free(arena);
    cuda->Free(weights);
    cuda->UnloadModule(module);
  }

  Status Prepare();
  Status Bind(std::uint32_t port_id, ai_port_kind kind,
              const ai_tensor_view* view);
  Status Launch(void* stream);
};

namespace {

ai_status Record(ai_session* session, const Status& status) {
  if (status.ok()) return AI_STATUS_OK;
  if (session != nullptr) session->last_error = status.message();
  return Record(status);
}

template <typename Function>
ai_status GuardSession(ai_session* session, Function&& function) noexcept {
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return Record(session,
                  Error(StatusCode::kOutOfMemory, "host allocation failed"));
  } catch (const std::exception& error) {
    return Record(session, Error(StatusCode::kInvalidState,
                                 std::string("unexpected runtime exception: ") +
                                     error.what()));
  } catch (...) {
    return Record(session, Error(StatusCode::kInvalidState,
                                 "unknown runtime exception"));
  }
}

Status ProbeCuda(ai_runtime* runtime) {
  if (runtime->cuda_arch != 0 && runtime->cuda_driver_version != 0) {
    return Status::Ok();
  }
  void* cuda = OpenLibrary({"libcuda.so.1", "libcuda.so"});
  if (cuda == nullptr) {
    return Error(StatusCode::kCudaError,
                 "cannot load the CUDA Driver library");
  }
  using Init = int (*)(unsigned);
  using DeviceGet = int (*)(int*, int);
  using AttributeGet = int (*)(int*, int, int);
  using DriverVersionGet = int (*)(int*);
  auto init = reinterpret_cast<Init>(dlsym(cuda, "cuInit"));
  auto device_get = reinterpret_cast<DeviceGet>(dlsym(cuda, "cuDeviceGet"));
  auto attribute_get =
      reinterpret_cast<AttributeGet>(dlsym(cuda, "cuDeviceGetAttribute"));
  auto version_get =
      reinterpret_cast<DriverVersionGet>(dlsym(cuda, "cuDriverGetVersion"));
  int device = 0;
  int major = 0;
  int minor = 0;
  int version = 0;
  const bool valid = init != nullptr && device_get != nullptr &&
                     attribute_get != nullptr && version_get != nullptr &&
                     init(0) == 0 && device_get(&device, 0) == 0 &&
                     attribute_get(&major, 75, device) == 0 &&
                     attribute_get(&minor, 76, device) == 0 &&
                     version_get(&version) == 0;
  dlclose(cuda);
  if (!valid) {
    return Error(StatusCode::kCudaError,
                 "cannot query CUDA device 0 and Driver version");
  }
  if (runtime->cuda_arch == 0) {
    runtime->cuda_arch = static_cast<std::uint32_t>(major * 10 + minor);
  }
  if (runtime->cuda_driver_version == 0) {
    runtime->cuda_driver_version = static_cast<std::uint32_t>(version);
  }
  return Status::Ok();
}

StatusOr<std::uint32_t> ProbeCudaRuntime() {
  void* cudart = OpenLibrary({"libcudart.so.12", "libcudart.so"});
  if (cudart == nullptr) {
    return Error(StatusCode::kIncompatibleAbi,
                 "AIM requires CUDA Runtime but its library is unavailable");
  }
  using RuntimeVersionGet = int (*)(int*);
  auto version_get = reinterpret_cast<RuntimeVersionGet>(
      dlsym(cudart, "cudaRuntimeGetVersion"));
  int version = 0;
  const bool valid = version_get != nullptr && version_get(&version) == 0;
  dlclose(cudart);
  if (!valid) {
    return Error(StatusCode::kIncompatibleAbi,
                 "cannot query required CUDA Runtime version");
  }
  return static_cast<std::uint32_t>(version);
}

StatusOr<std::uint32_t> ProbeProvider(std::uint32_t provider_id) {
  if (provider_id == AI_PROVIDER_CUBLASLT) {
    if (void* library = OpenLibrary({"libcublasLt.so.12"})) {
      dlclose(library);
      return std::uint32_t{12};
    }
    if (void* library = OpenLibrary({"libcublasLt.so.11"})) {
      dlclose(library);
      return std::uint32_t{11};
    }
  } else if (provider_id == AI_PROVIDER_CUDNN) {
    if (void* library = OpenLibrary({"libcudnn.so.9"})) {
      dlclose(library);
      return std::uint32_t{9};
    }
    if (void* library = OpenLibrary({"libcudnn.so.8"})) {
      dlclose(library);
      return std::uint32_t{8};
    }
  }
  return Error(StatusCode::kIncompatibleAbi,
               "required provider ABI is unavailable: provider " +
                   std::to_string(provider_id));
}

StatusOr<std::uint32_t> ProviderVersion(ai_runtime* runtime,
                                        std::uint32_t provider_id) {
  for (const auto& provider : runtime->provider_abis) {
    if (provider.provider_id == provider_id) return provider.abi_version;
  }
  auto probed = ProbeProvider(provider_id);
  if (!probed.ok()) return probed.status();
  runtime->provider_abis.push_back({provider_id, probed.value()});
  return probed.value();
}

Status CheckCompatibility(ai_runtime* runtime, const ai_model* model) {
  const CompatibilityHeader& compatibility = *model->compatibility;
  if (!RangeContains(runtime->cuda_driver_version,
                     compatibility.cuda_driver_min,
                     compatibility.cuda_driver_max)) {
    return Error(StatusCode::kIncompatibleAbi,
                 "CUDA Driver version is outside the AIM compatibility range");
  }
  if (compatibility.cuda_runtime_min != 0) {
    if (runtime->cuda_runtime_version == 0) {
      auto version = ProbeCudaRuntime();
      if (!version.ok()) return version.status();
      runtime->cuda_runtime_version = version.value();
    }
    if (!RangeContains(runtime->cuda_runtime_version,
                       compatibility.cuda_runtime_min,
                       compatibility.cuda_runtime_max)) {
      return Error(
          StatusCode::kIncompatibleAbi,
          "CUDA Runtime version is outside the AIM compatibility range");
    }
  }
  for (std::uint32_t index = 0; index < compatibility.provider_count;
       ++index) {
    const ProviderRequirement& requirement = model->providers[index];
    auto version = ProviderVersion(runtime, requirement.provider_id);
    if (!version.ok()) return version.status();
    if (!RangeContains(version.value(), requirement.abi_min,
                       requirement.abi_max)) {
      return Error(StatusCode::kIncompatibleAbi,
                   "provider ABI is outside the AIM compatibility range: " +
                       std::to_string(requirement.provider_id));
    }
  }
  return Status::Ok();
}

Status ValidateCompatibility(ai_model* model) {
  const Header& header = *model->header;
#ifdef AGINFER_VERIFY_AIM_CHECKSUMS
  if (!CheckHash(model->data + header.compatibility_offset,
                 static_cast<std::size_t>(header.compatibility_size),
                 header.compatibility_sha)) {
    return Error(StatusCode::kCorruptPackage,
                 "AIM compatibility table checksum mismatch");
  }
#endif
  if (header.compatibility_size < sizeof(CompatibilityHeader)) {
    return Error(StatusCode::kCorruptPackage,
                 "AIM compatibility table is smaller than its header");
  }
  model->compatibility = reinterpret_cast<const CompatibilityHeader*>(
      model->data + header.compatibility_offset);
  const CompatibilityHeader& compatibility = *model->compatibility;
  if (!std::equal(kCompatibilityMagic.begin(), kCompatibilityMagic.end(),
                  compatibility.magic) ||
      compatibility.schema_major != 1 ||
      compatibility.schema_minor > 0 ||
      compatibility.header_size != sizeof(CompatibilityHeader) ||
      compatibility.provider_count > kMaxProviderRequirements ||
      compatibility.flags != 0 ||
      !AllZero(compatibility.reserved, sizeof(compatibility.reserved)) ||
      compatibility.providers_offset != sizeof(CompatibilityHeader) ||
      compatibility.section_size != header.compatibility_size) {
    return Error(StatusCode::kCorruptPackage,
                 "invalid AIM compatibility header or version");
  }
  const std::uint64_t provider_bytes =
      static_cast<std::uint64_t>(compatibility.provider_count) *
      sizeof(ProviderRequirement);
  if (provider_bytes !=
      header.compatibility_size - sizeof(CompatibilityHeader)) {
    return Error(StatusCode::kCorruptPackage,
                 "AIM compatibility provider table has invalid bounds");
  }
  if (!VersionRangeValid(compatibility.cuda_driver_min,
                         compatibility.cuda_driver_max, true) ||
      !VersionRangeValid(compatibility.cuda_runtime_min,
                         compatibility.cuda_runtime_max, false)) {
    return Error(StatusCode::kCorruptPackage,
                 "AIM compatibility table has an invalid version range");
  }
  model->providers = reinterpret_cast<const ProviderRequirement*>(
      reinterpret_cast<const std::uint8_t*>(model->compatibility) +
      compatibility.providers_offset);
  std::uint32_t previous_id = 0;
  for (std::uint32_t index = 0; index < compatibility.provider_count;
       ++index) {
    const ProviderRequirement& provider = model->providers[index];
    if (provider.provider_id <= previous_id ||
        !VersionRangeValid(provider.abi_min, provider.abi_max, true) ||
        provider.flags != 0 ||
        !AllZero(provider.reserved, sizeof(provider.reserved))) {
      return Error(StatusCode::kCorruptPackage,
                   "invalid, duplicate, or unsorted AIM provider record");
    }
    previous_id = provider.provider_id;
  }
  return Status::Ok();
}

Status LoadModel(const char* path, std::unique_ptr<ai_model>* output) {
  if (path == nullptr || path[0] == '\0' || output == nullptr) {
    return Error(StatusCode::kInvalidArgument,
                 "AIM path and model output must be non-null");
  }
  auto model = std::make_unique<ai_model>();
  model->fd = open(path, O_RDONLY | O_CLOEXEC);
  if (model->fd < 0) {
    return Error(errno == ENOENT ? StatusCode::kNotFound
                                 : StatusCode::kIoError,
                 "cannot open AIM: " + std::string(std::strerror(errno)));
  }
  struct stat attributes {};
  if (fstat(model->fd, &attributes) != 0 ||
      attributes.st_size < static_cast<off_t>(sizeof(Header))) {
    return Error(StatusCode::kCorruptPackage,
                 "AIM is smaller than its fixed header");
  }
  model->size = static_cast<std::size_t>(attributes.st_size);
  void* mapped =
      mmap(nullptr, model->size, PROT_READ, MAP_PRIVATE, model->fd, 0);
  if (mapped == MAP_FAILED) {
    return Error(StatusCode::kIoError, "cannot mmap AIM");
  }
  model->data = static_cast<const std::uint8_t*>(mapped);
  model->header = reinterpret_cast<const Header*>(mapped);
  const Header& header = *model->header;
  if (!std::equal(kMagic.begin(), kMagic.end(), header.magic)) {
    return Error(StatusCode::kCorruptPackage, "bad AIM magic");
  }
  if (header.schema_major != 2 || header.schema_minor > 0 ||
      header.header_size != sizeof(Header) ||
      header.endian_tag != kEndianTag) {
    return Error(StatusCode::kCorruptPackage,
                 "unsupported AIM schema, header, or byte order");
  }
  if (header.runtime_abi != AI_RUNTIME_ABI_VERSION) {
    return Error(StatusCode::kIncompatibleAbi,
                 "AIM Runtime ABI mismatch");
  }
  if (header.platform != HostPlatformId()) {
    return Error(StatusCode::kIncompatiblePlatform,
                 "AIM platform does not match host " +
                     std::string(HostPlatformName()));
  }
  const std::uint64_t expected_directory_size =
      static_cast<std::uint64_t>(header.variant_count) * sizeof(Variant);
  if (header.file_size != model->size || header.variant_count == 0 ||
      !RegionValid(model->size, header.manifest_offset,
                   header.manifest_size) ||
      !RegionValid(model->size, header.graph_offset, header.graph_size) ||
      !RegionValid(model->size, header.tensor_offset, header.tensor_size) ||
      !RegionValid(model->size, header.compatibility_offset,
                   header.compatibility_size) ||
      !RegionValid(model->size, header.directory_offset,
                   header.directory_size) ||
      header.directory_size != expected_directory_size) {
    return Error(StatusCode::kCorruptPackage,
                 "invalid AIM section bounds");
  }
#ifdef AGINFER_VERIFY_AIM_CHECKSUMS
  // Full content scans belong to verification/debug builds. Bounds and ABI
  // validation below remain mandatory in deployment builds.
  if (!CheckHash(model->data + header.directory_offset,
                 static_cast<std::size_t>(header.directory_size),
                 header.directory_sha) ||
      !CheckHash(model->data + header.manifest_offset,
                 static_cast<std::size_t>(header.manifest_size),
                 header.manifest_sha) ||
      !CheckHash(model->data + header.graph_offset,
                 static_cast<std::size_t>(header.graph_size),
                 header.graph_sha)) {
    return Error(StatusCode::kCorruptPackage,
                 "AIM metadata checksum mismatch");
  }
  const auto file_hash = FileHash(model->data, model->size);
  if (!HashEquals(file_hash.data(), header.file_sha)) {
    return Error(StatusCode::kCorruptPackage, "AIM file checksum mismatch");
  }
#endif
  if (!AllZero(header.reserved, sizeof(header.reserved))) {
    return Error(StatusCode::kCorruptPackage,
                 "non-zero AIM reserved bytes");
  }
  Status status = ValidateCompatibility(model.get());
  if (!status.ok()) return status;

  model->variants = reinterpret_cast<const Variant*>(
      model->data + header.directory_offset);
  for (std::uint32_t index = 0; index < header.variant_count; ++index) {
    const Variant& variant = model->variants[index];
    if (!ArchMatchesPlatform(header.platform, variant.arch) ||
        variant.flags != 0 ||
        !AllZero(variant.reserved, sizeof(variant.reserved)) ||
        variant.kernel_offset % 256 != 0 ||
        variant.weight_offset % 256 != 0 ||
        variant.plan_offset % 256 != 0 ||
        !RegionValid(model->size, variant.kernel_offset,
                     variant.kernel_size) ||
        !RegionValid(model->size, variant.weight_offset,
                     variant.weight_size) ||
        !RegionValid(model->size, variant.plan_offset, variant.plan_size) ||
        variant.weight_size > std::numeric_limits<std::size_t>::max()) {
      return Error(StatusCode::kCorruptPackage, "invalid variant or payload bounds");
    }
#ifdef AGINFER_VERIFY_AIM_CHECKSUMS
    if (!CheckHash(model->data + variant.kernel_offset,
                   static_cast<std::size_t>(variant.kernel_size),
                   variant.kernel_sha) ||
        !CheckHash(model->data + variant.weight_offset,
                   static_cast<std::size_t>(variant.weight_size),
                   variant.weight_sha) ||
        !CheckHash(model->data + variant.plan_offset,
                   static_cast<std::size_t>(variant.plan_size),
                   variant.plan_sha)) {
      return Error(StatusCode::kCorruptPackage,
                   "invalid variant or payload checksum");
    }
#endif
    if (ContainsPtx(model->data + variant.kernel_offset,
                    static_cast<std::size_t>(variant.kernel_size))) {
      return Error(StatusCode::kCorruptPackage,
                   "PTX in an AIM kernel bundle is forbidden");
    }
    if (!IsExactCubin(model->data + variant.kernel_offset,
                      static_cast<std::size_t>(variant.kernel_size),
                      variant.arch)) {
      return Error(StatusCode::kCorruptPackage,
                   "kernel bundle is not an exact-architecture NVIDIA CUBIN");
    }
    for (std::uint32_t previous = 0; previous < index; ++previous) {
      if (model->variants[previous].arch == variant.arch) {
        return Error(StatusCode::kCorruptPackage,
                     "duplicate CUDA architecture variant");
      }
    }
  }
  *output = std::move(model);
  return Status::Ok();
}

Status CreateSession(ai_runtime* runtime, ai_model* model,
                     const ai_session_options* options,
                     std::unique_ptr<ai_session>* output) {
  if (runtime == nullptr || model == nullptr || output == nullptr) {
    return Error(StatusCode::kInvalidArgument,
                 "runtime, model, and session output must be non-null");
  }
  ai_session_options selected_options{};
  ai_session_options_init(&selected_options);
  if (options != nullptr) {
    Status status = ValidateStruct(options, sizeof(ai_session_options),
                                   "ai_session_options");
    if (!status.ok()) return status;
    if (options->flags != 0) {
      return Error(StatusCode::kInvalidArgument,
                   "ai_session_options has unknown flags");
    }
    selected_options = *options;
  }
  const Variant* selected = nullptr;
  for (std::uint32_t index = 0; index < model->header->variant_count;
       ++index) {
    if (model->variants[index].arch == runtime->cuda_arch) {
      selected = &model->variants[index];
      break;
    }
  }
  if (selected == nullptr) {
    return Error(StatusCode::kIncompatibleArchitecture,
                 "AIM has no exact sm" + std::to_string(runtime->cuda_arch) +
                     " variant; fallback and PTX JIT are forbidden");
  }
  Status status = CheckCompatibility(runtime, model);
  if (!status.ok()) return status;

  auto session = std::make_unique<ai_session>();
  session->runtime = runtime;
  session->model = model;
  session->variant = selected;
  if (selected->plan_size >= 8 &&
      std::memcmp(model->data + selected->plan_offset, "AIMEXE2", 8) == 0) {
    if (selected_options.profile_index != AI_DEFAULT_PROFILE && selected_options.profile_index != 0)
      return Error(StatusCode::kInvalidArgument, "executable v2 has exactly one profile");
    aginfer::internal::ParsedExecutablePlan executable_plan;
    status = aginfer::internal::ParseExecutablePlan(model->data + selected->plan_offset,
        selected->plan_size, selected->arch, selected->weight_size, &executable_plan);
    if (!status.ok()) return status;
    session->executable = std::make_unique<aginfer::internal::ExecutableSession>(
        executable_plan, model->data + selected->kernel_offset, selected->kernel_size,
        model->data + selected->weight_offset);
    *output = std::move(session);
    return Status::Ok();
  }
  status = aginfer::internal::ParsePlan(
      model->data + selected->plan_offset,
      static_cast<std::size_t>(selected->plan_size), selected->arch,
      selected->weight_size, &session->plan);
  if (!status.ok()) return status;
  const std::uint32_t profile_index =
      selected_options.profile_index == AI_DEFAULT_PROFILE
          ? 0
          : selected_options.profile_index;
  if (profile_index >= session->plan.header->profile_count) {
    return Error(StatusCode::kInvalidArgument,
                 "profile index is not present in the static execution plan");
  }
  session->profile = &session->plan.profiles[profile_index];
  session->tensor_bindings.resize(session->plan.header->tensor_count);
  std::uint32_t max_arguments = 0;
  for (std::uint32_t index = 0; index < session->profile->launch_count;
       ++index) {
    max_arguments = std::max(
        max_arguments,
        session->plan
            .launches[session->profile->first_launch + index]
            .argument_count);
  }
  session->argument_values.resize(max_arguments);
  session->kernel_parameters.resize(max_arguments);
  *output = std::move(session);
  return Status::Ok();
}

const aginfer::internal::PlanTensor* FindPort(
    const ai_session* session, ai_port_kind kind, std::uint32_t port_id,
    std::uint32_t* global_index = nullptr) {
  for (std::uint32_t index = 0; index < session->profile->tensor_count;
       ++index) {
    const std::uint32_t candidate_index =
        session->profile->first_tensor + index;
    const auto& tensor = session->plan.tensors[candidate_index];
    if (tensor.io_kind == kind && tensor.port_id == port_id) {
      if (global_index != nullptr) *global_index = candidate_index;
      return &tensor;
    }
  }
  return nullptr;
}

}  // namespace

Status ai_session::Prepare() {
  if (executable) return executable->Prepare();
  if (prepared) return cuda->MakeCurrent();
  if (cuda != nullptr) {
    return Error(StatusCode::kInvalidState,
                 "session Prepare previously failed; recreate the session");
  }
  auto created = aginfer::internal::CudaDriver::Create(variant->arch);
  if (!created.ok()) return created.status();
  cuda = std::make_unique<aginfer::internal::CudaDriver>(
      std::move(created).value());
  auto loaded_module = cuda->LoadModule(model->data + variant->kernel_offset);
  if (!loaded_module.ok()) return loaded_module.status();
  module = loaded_module.value();

  auto weight_allocation = cuda->Allocate(variant->weight_size);
  if (!weight_allocation.ok()) return weight_allocation.status();
  weights = weight_allocation.value();
  Status status = cuda->CopyHostToDevice(
      weights, model->data + variant->weight_offset,
      static_cast<std::size_t>(variant->weight_size));
  if (!status.ok()) return status;
  auto arena_allocation = cuda->Allocate(plan.header->arena_bytes);
  if (!arena_allocation.ok()) return arena_allocation.status();
  arena = arena_allocation.value();
  auto workspace_allocation = cuda->Allocate(plan.header->workspace_bytes);
  if (!workspace_allocation.ok()) return workspace_allocation.status();
  workspace = workspace_allocation.value();

  functions.clear();
  functions.reserve(profile->launch_count);
  for (std::uint32_t index = 0; index < profile->launch_count; ++index) {
    const auto& launch = plan.launches[profile->first_launch + index];
    auto function = cuda->GetFunction(
        module, std::string(plan.String(launch.kernel_name_offset)));
    if (!function.ok()) return function.status();
    functions.push_back(function.value());
  }
  prepared = true;
  return Status::Ok();
}

Status ai_session::Bind(std::uint32_t port_id, ai_port_kind kind,
                        const ai_tensor_view* view) {
  Status status = ValidateStruct(view, sizeof(ai_tensor_view),
                                 "ai_tensor_view");
  if (!status.ok()) return status;
  if (kind != AI_PORT_INPUT && kind != AI_PORT_OUTPUT) {
    return Error(StatusCode::kInvalidArgument, "invalid port kind");
  }
  if (executable) return executable->Bind(port_id, kind, *view);
  if (view->flags != 0 ||
      (view->dtype < AI_DTYPE_F32 || view->dtype > AI_DTYPE_I32) ||
      (view->location != AI_MEMORY_HOST &&
       view->location != AI_MEMORY_DEVICE)) {
    return Error(StatusCode::kInvalidArgument,
                 "tensor view has an invalid enum or flags");
  }
  std::uint32_t global_index = 0;
  const auto* expected = FindPort(this, kind, port_id, &global_index);
  if (expected == nullptr) {
    return Error(StatusCode::kInvalidArgument,
                 "numeric port ID is not present in the selected profile");
  }
  if (view->dtype != expected->dtype ||
      view->location != expected->location || view->rank != expected->rank) {
    return Error(StatusCode::kInvalidArgument,
                 "dtype, location, or rank does not match the port contract");
  }
  if (view->data == nullptr || view->shape == nullptr ||
      view->stride == nullptr || view->byte_size < expected->byte_size) {
    return Error(StatusCode::kInvalidArgument,
                 "tensor buffer, shape, stride, or byte size is invalid");
  }
  for (std::uint32_t dimension = 0; dimension < expected->rank;
       ++dimension) {
    if (view->shape[dimension] != expected->shape[dimension] ||
        view->stride[dimension] != expected->stride[dimension]) {
      return Error(StatusCode::kInvalidArgument,
                   "shape or stride is outside the selected profile");
    }
  }
  tensor_bindings[global_index] = {true, view->data};
  return Status::Ok();
}

Status ai_session::Launch(void* stream) {
  if (executable) return executable->Enqueue(stream);
  if (!prepared) {
    return Error(StatusCode::kInvalidState,
                 "session must be prepared before Enqueue");
  }
  for (std::uint32_t index = 0; index < profile->tensor_count; ++index) {
    if (!tensor_bindings[profile->first_tensor + index].bound) {
      return Error(StatusCode::kInvalidState,
                   "all selected-profile ports must be bound before Enqueue");
    }
  }
  for (std::uint32_t launch_index = 0;
       launch_index < profile->launch_count; ++launch_index) {
    const auto& launch =
        plan.launches[profile->first_launch + launch_index];
    for (std::uint32_t index = 0; index < launch.argument_count; ++index) {
      const auto& argument = plan.arguments[launch.first_argument + index];
      switch (static_cast<aginfer::internal::PlanArgKind>(argument.kind)) {
        case aginfer::internal::PlanArgKind::kTensor:
          argument_values[index] = static_cast<std::uint64_t>(
              reinterpret_cast<std::uintptr_t>(
                  tensor_bindings[argument.index].data));
          break;
        case aginfer::internal::PlanArgKind::kScalarU32:
        case aginfer::internal::PlanArgKind::kScalarF32:
          argument_values[index] = argument.value;
          break;
        case aginfer::internal::PlanArgKind::kArenaOffset:
          argument_values[index] = arena + argument.offset;
          break;
        case aginfer::internal::PlanArgKind::kWeightsOffset:
          argument_values[index] = weights + argument.offset;
          break;
      }
      kernel_parameters[index] = &argument_values[index];
    }
    Status status = cuda->Launch(
        functions[launch_index], launch.grid, launch.block,
        launch.shared_bytes, stream, kernel_parameters.data());
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

extern "C" {

void ai_provider_abi_init(ai_provider_abi* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
}

void ai_runtime_options_init(ai_runtime_options* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
  value->required_runtime_abi = AI_RUNTIME_ABI_VERSION;
}

void ai_session_options_init(ai_session_options* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
  value->profile_index = AI_DEFAULT_PROFILE;
}

void ai_tensor_view_init(ai_tensor_view* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
}

void ai_port_info_init(ai_port_info* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
}

void ai_target_info_init(ai_target_info* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
}

void ai_workspace_info_init(ai_workspace_info* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value);
  value->struct_version = AI_STRUCT_VERSION_1;
}

ai_status ai_runtime_create(const ai_runtime_options* options,
                            ai_runtime** runtime_out) {
  return Guard([&]() -> ai_status {
    if (runtime_out == nullptr) {
      return Fail(StatusCode::kInvalidArgument,
                  "runtime output must not be null");
    }
    *runtime_out = nullptr;
    ai_runtime_options selected{};
    ai_runtime_options_init(&selected);
    if (options != nullptr) {
      Status status = ValidateStruct(options, sizeof(ai_runtime_options),
                                     "ai_runtime_options");
      if (!status.ok()) return Record(status);
      if (options->required_runtime_abi != AI_RUNTIME_ABI_VERSION) {
        return Fail(StatusCode::kIncompatibleAbi,
                    "requested Runtime ABI does not match this library");
      }
      if (options->flags != 0 ||
          options->provider_abi_count > kMaxProviderRequirements ||
          (options->provider_abi_count != 0 &&
           options->provider_abis == nullptr)) {
        return Fail(StatusCode::kInvalidArgument,
                    "runtime options contain invalid flags or provider table");
      }
      selected = *options;
    }
    auto runtime = std::make_unique<ai_runtime>();
    runtime->cuda_arch = selected.cuda_arch_override;
    runtime->cuda_driver_version =
        selected.cuda_driver_version_override;
    runtime->cuda_runtime_version =
        selected.cuda_runtime_version_override;
    for (std::uint32_t index = 0; index < selected.provider_abi_count;
         ++index) {
      const ai_provider_abi& provider = selected.provider_abis[index];
      Status status = ValidateStruct(&provider, sizeof(ai_provider_abi),
                                     "ai_provider_abi");
      if (!status.ok()) return Record(status);
      if (provider.provider_id == 0 || provider.abi_version == 0) {
        return Fail(StatusCode::kInvalidArgument,
                    "provider ID and ABI version must be positive");
      }
      for (const auto& existing : runtime->provider_abis) {
        if (existing.provider_id == provider.provider_id) {
          return Fail(StatusCode::kInvalidArgument,
                      "duplicate runtime provider ABI ID");
        }
      }
      runtime->provider_abis.push_back(
          {provider.provider_id, provider.abi_version});
    }
    Status status = ProbeCuda(runtime.get());
    if (!status.ok()) return Record(status);
    *runtime_out = runtime.release();
    return AI_STATUS_OK;
  });
}

void ai_runtime_destroy(ai_runtime* runtime) { delete runtime; }

ai_status ai_model_load(const char* aim_path, ai_model** model_out) {
  return Guard([&]() -> ai_status {
    if (model_out == nullptr) {
      return Fail(StatusCode::kInvalidArgument,
                  "model output must not be null");
    }
    *model_out = nullptr;
    std::unique_ptr<ai_model> model;
    Status status = LoadModel(aim_path, &model);
    if (!status.ok()) return Record(status);
    *model_out = model.release();
    return AI_STATUS_OK;
  });
}

void ai_model_destroy(ai_model* model) { delete model; }

ai_status ai_session_create(ai_runtime* runtime, ai_model* model,
                            const ai_session_options* options,
                            ai_session** session_out) {
  return Guard([&]() -> ai_status {
    if (session_out == nullptr) {
      return Fail(StatusCode::kInvalidArgument,
                  "session output must not be null");
    }
    *session_out = nullptr;
    std::unique_ptr<ai_session> session;
    Status status = CreateSession(runtime, model, options, &session);
    if (!status.ok()) return Record(status);
    *session_out = session.release();
    return AI_STATUS_OK;
  });
}

void ai_session_destroy(ai_session* session) { delete session; }

ai_status ai_session_prepare(ai_session* session) {
  if (session == nullptr) {
    return Fail(StatusCode::kInvalidArgument,
                "session must not be null");
  }
  return GuardSession(session,
                      [&]() { return Record(session, session->Prepare()); });
}

ai_status ai_session_get_target_info(const ai_session* session,
                                     ai_target_info* info) {
  return Guard([&]() -> ai_status {
    if (session == nullptr) {
      return Fail(StatusCode::kInvalidArgument,
                  "session must not be null");
    }
    Status status =
        ValidateStruct(info, sizeof(ai_target_info), "ai_target_info");
    if (!status.ok()) return Record(status);
    const std::uint32_t caller_size = info->struct_size;
    ai_target_info_init(info);
    info->struct_size = caller_size;
    info->platform = session->model->header->platform;
    info->cuda_arch = session->variant->arch;
    info->runtime_abi = session->model->header->runtime_abi;
    return AI_STATUS_OK;
  });
}

ai_status ai_session_get_workspace_info(const ai_session* session,
                                        ai_workspace_info* info) {
  return Guard([&]() -> ai_status {
    if (session == nullptr) {
      return Fail(StatusCode::kInvalidArgument,
                  "session must not be null");
    }
    Status status = ValidateStruct(info, sizeof(ai_workspace_info),
                                   "ai_workspace_info");
    if (!status.ok()) return Record(status);
    const std::uint32_t caller_size = info->struct_size;
    ai_workspace_info_init(info);
    info->struct_size = caller_size;
    info->arena_bytes = session->executable ? session->executable->plan().arena_bytes : session->plan.header->arena_bytes;
    info->workspace_bytes = session->executable ? session->executable->plan().workspace_bytes : session->plan.header->workspace_bytes;
    return AI_STATUS_OK;
  });
}

ai_status ai_session_get_port_count(const ai_session* session,
                                    ai_port_kind kind,
                                    std::uint32_t* count) {
  return Guard([&]() -> ai_status {
    if (session == nullptr || count == nullptr ||
        (kind != AI_PORT_INPUT && kind != AI_PORT_OUTPUT)) {
      return Fail(StatusCode::kInvalidArgument,
                  "session, port kind, or count output is invalid");
    }
    *count = 0;
    if (session->executable) {
      *count = session->executable->PortCount(kind);
      return AI_STATUS_OK;
    }
    for (std::uint32_t index = 0; index < session->profile->tensor_count;
         ++index) {
      const auto& tensor = session->plan.tensors[
          session->profile->first_tensor + index];
      if (tensor.io_kind == kind) ++*count;
    }
    return AI_STATUS_OK;
  });
}

ai_status ai_session_get_port_info(const ai_session* session,
                                   ai_port_kind kind, std::uint32_t index,
                                   ai_port_info* info) {
  return Guard([&]() -> ai_status {
    if (session == nullptr ||
        (kind != AI_PORT_INPUT && kind != AI_PORT_OUTPUT)) {
      return Fail(StatusCode::kInvalidArgument,
                  "session or port kind is invalid");
    }
    Status status =
        ValidateStruct(info, sizeof(ai_port_info), "ai_port_info");
    if (!status.ok()) return Record(status);
    if (session->executable)
      return Record(session->executable->PortInfo(kind, index, info));
    const aginfer::internal::PlanTensor* selected = nullptr;
    std::uint32_t matched = 0;
    for (std::uint32_t candidate = 0;
         candidate < session->profile->tensor_count; ++candidate) {
      const auto& tensor = session->plan.tensors[
          session->profile->first_tensor + candidate];
      if (tensor.io_kind != kind) continue;
      if (matched++ == index) {
        selected = &tensor;
        break;
      }
    }
    if (selected == nullptr) {
      return Fail(StatusCode::kInvalidArgument,
                  "port index is outside the selected profile");
    }
    const std::uint32_t caller_size = info->struct_size;
    ai_port_info_init(info);
    info->struct_size = caller_size;
    info->port_id = selected->port_id;
    info->kind = kind;
    info->dtype = selected->dtype;
    info->location = selected->location;
    info->rank = selected->rank;
    info->byte_size = selected->byte_size;
    std::copy_n(selected->shape, selected->rank, info->shape);
    std::copy_n(selected->stride, selected->rank, info->stride);
    info->diagnostic_name = session->plan.String(selected->name_offset).data();
    return AI_STATUS_OK;
  });
}

ai_status ai_session_bind_input(ai_session* session, std::uint32_t port_id,
                                const ai_tensor_view* view) {
  if (session == nullptr) {
    return Fail(StatusCode::kInvalidArgument,
                "session must not be null");
  }
  return GuardSession(session, [&]() {
    return Record(session, session->Bind(port_id, AI_PORT_INPUT, view));
  });
}

ai_status ai_session_bind_output(ai_session* session, std::uint32_t port_id,
                                 const ai_tensor_view* view) {
  if (session == nullptr) {
    return Fail(StatusCode::kInvalidArgument,
                "session must not be null");
  }
  return GuardSession(session, [&]() {
    return Record(session, session->Bind(port_id, AI_PORT_OUTPUT, view));
  });
}

ai_status ai_session_enqueue(ai_session* session, void* cuda_stream) {
  if (session == nullptr) {
    return Fail(StatusCode::kInvalidArgument,
                "session must not be null");
  }
  return GuardSession(session, [&]() {
    return Record(session, session->Launch(cuda_stream));
  });
}

void ai_execution_info_init(ai_execution_info* value) {
  if (value == nullptr) return;
  std::memset(value, 0, sizeof(*value));
  value->struct_size = sizeof(*value); value->struct_version = AI_STRUCT_VERSION_1;
}

ai_status ai_session_get_execution_info(const ai_session* session, std::uint32_t provider,
                                       ai_execution_info* info) {
  return Guard([&]() -> ai_status {
    if (session == nullptr || !session->executable)
      return Fail(StatusCode::kInvalidArgument, "execution ledger requires an executable v2 session");
    auto status = ValidateStruct(info, sizeof(ai_execution_info), "ai_execution_info");
    if (!status.ok()) return Record(status);
    auto caller_size = info->struct_size;
    ai_execution_info_init(info); info->struct_size = caller_size;
    info->enqueues = session->executable->enqueues();
    info->commands_submitted = session->executable->submitted(provider);
    const auto& plan = session->executable->plan();
    if (provider == 0) info->commands_per_enqueue = plan.command_stream.command_count;
    else {
      for (std::uint32_t i = 0; i < plan.provider_count; ++i) {
        aginfer::internal::ExecutableProviderView p; plan.ReadProvider(i, &p);
        if (p.provider_id == provider) info->commands_per_enqueue = p.command_count;
      }
      if (!info->commands_per_enqueue) return Fail(StatusCode::kInvalidArgument, "unknown executable provider ID");
    }
    return AI_STATUS_OK;
  });
}

const char* ai_session_last_error(const ai_session* session) {
  return session == nullptr ? "session is null" : session->last_error.c_str();
}

const char* ai_last_error(void) { return g_last_error.c_str(); }

const char* ai_status_name(ai_status status) {
  switch (status) {
    case AI_STATUS_OK:
      return "OK";
    case AI_STATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case AI_STATUS_NOT_FOUND:
      return "NOT_FOUND";
    case AI_STATUS_IO_ERROR:
      return "IO_ERROR";
    case AI_STATUS_CORRUPT_PACKAGE:
      return "CORRUPT_PACKAGE";
    case AI_STATUS_INCOMPATIBLE_PLATFORM:
      return "INCOMPATIBLE_PLATFORM";
    case AI_STATUS_INCOMPATIBLE_ARCHITECTURE:
      return "INCOMPATIBLE_ARCHITECTURE";
    case AI_STATUS_INCOMPATIBLE_ABI:
      return "INCOMPATIBLE_ABI";
    case AI_STATUS_CUDA_ERROR:
      return "CUDA_ERROR";
    case AI_STATUS_OUT_OF_MEMORY:
      return "OUT_OF_MEMORY";
    case AI_STATUS_INVALID_STATE:
      return "INVALID_STATE";
    default:
      return "UNKNOWN";
  }
}

}  // extern "C"
