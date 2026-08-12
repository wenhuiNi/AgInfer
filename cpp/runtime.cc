#include "aginfer/runtime.h"

#include "sha256.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace aginfer {
namespace {
#pragma pack(push, 1)
struct Header {
  char magic[8]; std::uint16_t schema_major; std::uint16_t schema_minor;
  std::uint32_t header_size; std::uint32_t runtime_abi; std::uint32_t platform;
  std::uint32_t endian_tag; std::uint32_t variant_count;
  std::uint64_t file_size; std::uint64_t manifest_offset; std::uint64_t manifest_size;
  std::uint64_t graph_offset; std::uint64_t graph_size; std::uint64_t tensor_offset;
  std::uint64_t tensor_size; std::uint64_t directory_offset; std::uint64_t directory_size;
  std::uint8_t directory_sha[32]; std::uint8_t file_sha[32]; std::uint8_t manifest_sha[32];
  std::uint8_t graph_sha[32]; std::uint8_t reserved[24];
};
struct Variant {
  std::uint32_t arch; std::uint32_t flags;
  std::uint64_t kernel_offset; std::uint64_t kernel_size; std::uint64_t weight_offset;
  std::uint64_t weight_size; std::uint64_t plan_offset; std::uint64_t plan_size;
  std::uint8_t kernel_sha[32]; std::uint8_t weight_sha[32]; std::uint8_t plan_sha[32];
  std::uint8_t reserved[40];
};
#pragma pack(pop)
static_assert(sizeof(Header) == 256);
static_assert(sizeof(Variant) == 192);
constexpr std::array<char, 8> kMagic{'A','I','M','A','O','T','1','\0'};
constexpr std::uint32_t kEndianTag = 0x01020304;
constexpr std::size_t kFileHashOffset = 136;

Status Error(StatusCode code, const std::string& text) { return Status(code, text); }

bool RegionValid(std::size_t total, std::uint64_t offset, std::uint64_t size) {
  return offset >= sizeof(Header) && size > 0 && offset <= total && size <= total - offset;
}

std::array<std::uint8_t, 32> Hash(const std::uint8_t* data, std::size_t size) {
  internal::Sha256 hash; hash.Update(data, size); return hash.Final();
}
bool HashEquals(const std::uint8_t* actual, const std::uint8_t* expected) {
  unsigned difference = 0;
  for (std::size_t i = 0; i < 32; ++i) difference |= actual[i] ^ expected[i];
  return difference == 0;
}
bool CheckHash(const std::uint8_t* data, std::size_t size, const std::uint8_t expected[32]) {
  const auto digest = Hash(data, size); return HashEquals(digest.data(), expected);
}

std::array<std::uint8_t, 32> FileHash(const std::uint8_t* data, std::size_t size) {
  internal::Sha256 hash;
  hash.Update(data, kFileHashOffset);
  const std::array<std::uint8_t, 32> zeros{}; hash.Update(zeros.data(), zeros.size());
  hash.Update(data + kFileHashOffset + 32, size - kFileHashOffset - 32);
  return hash.Final();
}

std::string HostPlatform() {
#if defined(__linux__) && defined(__x86_64__)
  return "linux-x86_64-gnu";
#elif defined(__linux__) && defined(__aarch64__)
  return "linux-aarch64-sbsa";
#else
  return "unsupported";
#endif
}
std::uint32_t HostPlatformId() {
#if defined(__linux__) && defined(__x86_64__)
  return 1;
#elif defined(__linux__) && defined(__aarch64__)
  return 2;
#else
  return 0;
#endif
}
std::string ArchName(std::uint32_t arch) { return "sm" + std::to_string(arch); }
bool ArchMatchesPlatform(std::uint32_t platform, std::uint32_t arch) {
  return (platform == 1 && (arch == 89 || arch == 120)) || (platform == 2 && arch == 110);
}
bool ContainsPtx(const std::uint8_t* data, std::size_t size) {
  constexpr std::array<std::string_view, 4> markers{".version ", ".target sm_", ".entry ", ".visible .entry"};
  const std::string_view content(reinterpret_cast<const char*>(data), std::min<std::size_t>(size, 1024 * 1024));
  return std::any_of(markers.begin(), markers.end(), [&](std::string_view marker) { return content.find(marker) != std::string_view::npos; });
}
std::uint16_t LoadLittle16(const std::uint8_t* data) {
  return static_cast<std::uint16_t>(data[0]) | (static_cast<std::uint16_t>(data[1]) << 8);
}
std::uint32_t LoadLittle32(const std::uint8_t* data) {
  return static_cast<std::uint32_t>(data[0]) | (static_cast<std::uint32_t>(data[1]) << 8) |
         (static_cast<std::uint32_t>(data[2]) << 16) | (static_cast<std::uint32_t>(data[3]) << 24);
}
bool IsExactCubin(const std::uint8_t* data, std::size_t size, std::uint32_t arch) {
  constexpr std::array<std::uint8_t, 7> elf64_le{0x7f, 'E', 'L', 'F', 2, 1, 1};
  if (size < 64 || !std::equal(elf64_le.begin(), elf64_le.end(), data) || LoadLittle16(data + 18) != 190) return false;
  const std::uint32_t flags = LoadLittle32(data + 48);
  return (flags & 0xff) == arch || ((flags >> 16) & 0xff) == arch;
}

void* OpenLibrary(std::initializer_list<const char*> names) {
  for (const char* name : names) {
    if (void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL)) return handle;
  }
  return nullptr;
}

Status ProbeSystem(RuntimeOptions* options) {
  if (options->cuda_arch_override == 0 || options->cuda_driver_version_override == 0) {
    void* cuda = OpenLibrary({"libcuda.so.1", "libcuda.so"});
    if (cuda == nullptr) return Error(StatusCode::kCudaError, "cannot load the CUDA Driver library");
    using Init = int (*)(unsigned); using DeviceGet = int (*)(int*, int);
    using AttributeGet = int (*)(int*, int, int); using DriverVersionGet = int (*)(int*);
    auto init = reinterpret_cast<Init>(dlsym(cuda, "cuInit"));
    auto device_get = reinterpret_cast<DeviceGet>(dlsym(cuda, "cuDeviceGet"));
    auto attribute_get = reinterpret_cast<AttributeGet>(dlsym(cuda, "cuDeviceGetAttribute"));
    auto version_get = reinterpret_cast<DriverVersionGet>(dlsym(cuda, "cuDriverGetVersion"));
    int device = 0, major = 0, minor = 0, version = 0;
    const bool valid = init && device_get && attribute_get && version_get && init(0) == 0 &&
                       device_get(&device, 0) == 0 && attribute_get(&major, 75, device) == 0 &&
                       attribute_get(&minor, 76, device) == 0 && version_get(&version) == 0;
    dlclose(cuda);
    if (!valid) return Error(StatusCode::kCudaError, "cannot query CUDA device 0 and Driver version");
    if (options->cuda_arch_override == 0) options->cuda_arch_override = static_cast<std::uint32_t>(major * 10 + minor);
    if (options->cuda_driver_version_override == 0) options->cuda_driver_version_override = version;
  }
  if (options->cuda_runtime_version_override == 0) {
    void* cudart = OpenLibrary({"libcudart.so.12", "libcudart.so"});
    if (cudart == nullptr) return Error(StatusCode::kCudaError, "cannot load CUDA Runtime");
    using RuntimeVersionGet = int (*)(int*);
    auto version_get = reinterpret_cast<RuntimeVersionGet>(dlsym(cudart, "cudaRuntimeGetVersion"));
    int version = 0; const bool valid = version_get && version_get(&version) == 0; dlclose(cudart);
    if (!valid) return Error(StatusCode::kCudaError, "cannot query CUDA Runtime version");
    options->cuda_runtime_version_override = version;
  }
  if (options->cublaslt_abi_override == 0) {
    if (void* cublas = OpenLibrary({"libcublasLt.so.12"})) { options->cublaslt_abi_override = 12; dlclose(cublas); }
    else if (void* cublas = OpenLibrary({"libcublasLt.so.11"})) { options->cublaslt_abi_override = 11; dlclose(cublas); }
    else return Error(StatusCode::kCudaError, "cannot load a supported cuBLASLt ABI");
  }
  if (options->cudnn_abi_override == 0) {
    if (void* cudnn = OpenLibrary({"libcudnn.so.9"})) { options->cudnn_abi_override = 9; dlclose(cudnn); }
    else if (void* cudnn = OpenLibrary({"libcudnn.so.8"})) { options->cudnn_abi_override = 8; dlclose(cudnn); }
    else return Error(StatusCode::kCudaError, "cannot load a supported cuDNN ABI");
  }
  return Status::Ok();
}

StatusOr<std::uint64_t> JsonUnsigned(std::string_view json, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  auto position = json.find(needle);
  if (position == std::string_view::npos) return Error(StatusCode::kCorruptPackage, "manifest missing " + std::string(key));
  position = json.find(':', position + needle.size());
  if (position == std::string_view::npos) return Error(StatusCode::kCorruptPackage, "invalid manifest field " + std::string(key));
  ++position; while (position < json.size() && (json[position] == ' ' || json[position] == '\t')) ++position;
  std::uint64_t value = 0; bool found = false;
  while (position < json.size() && json[position] >= '0' && json[position] <= '9') {
    found = true; const auto digit = static_cast<unsigned>(json[position++] - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
      return Error(StatusCode::kCorruptPackage, "overflow in manifest field " + std::string(key));
    value = value * 10 + digit;
  }
  if (!found) return Error(StatusCode::kCorruptPackage, "invalid manifest integer " + std::string(key));
  return value;
}
}  // namespace

struct Runtime::Impl { RuntimeOptions options; };
struct Model::Impl {
  int fd = -1; std::size_t size = 0; const std::uint8_t* data = nullptr;
  const Header* header = nullptr; const Variant* variants = nullptr; std::string manifest;
  ~Impl() { if (data != nullptr) munmap(const_cast<std::uint8_t*>(data), size); if (fd >= 0) close(fd); }
};
struct Session::Impl {
  Runtime::Impl* runtime = nullptr; Model::Impl* model = nullptr; const Variant* variant = nullptr;
  std::string profile;
};

Runtime::Runtime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Runtime::Runtime(Runtime&&) noexcept = default; Runtime& Runtime::operator=(Runtime&&) noexcept = default; Runtime::~Runtime() = default;
Model::Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Model::Model(Model&&) noexcept = default; Model& Model::operator=(Model&&) noexcept = default; Model::~Model() = default;
Session::Session(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Session::Session(Session&&) noexcept = default; Session& Session::operator=(Session&&) noexcept = default; Session::~Session() = default;

StatusOr<Runtime> Runtime::Create(const RuntimeOptions& options) {
  if (options.required_runtime_abi != kRuntimeAbi)
    return Error(StatusCode::kIncompatibleAbi, "requested Runtime ABI does not match this library");
  auto impl = std::make_unique<Impl>(); impl->options = options;
  const Status probe = ProbeSystem(&impl->options);
  if (!probe.ok()) return probe;
  return Runtime(std::move(impl));
}

StatusOr<Model> Model::Load(const std::string& path) {
  auto impl = std::make_unique<Impl>();
  impl->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (impl->fd < 0) return Error(errno == ENOENT ? StatusCode::kNotFound : StatusCode::kIoError,
                                  "cannot open AIM: " + std::string(std::strerror(errno)));
  struct stat attributes {};
  if (fstat(impl->fd, &attributes) != 0 || attributes.st_size < static_cast<off_t>(sizeof(Header)))
    return Error(StatusCode::kCorruptPackage, "AIM is smaller than its fixed header");
  impl->size = static_cast<std::size_t>(attributes.st_size);
  void* mapped = mmap(nullptr, impl->size, PROT_READ, MAP_PRIVATE, impl->fd, 0);
  if (mapped == MAP_FAILED) return Error(StatusCode::kIoError, "cannot mmap AIM");
  impl->data = static_cast<const std::uint8_t*>(mapped); impl->header = reinterpret_cast<const Header*>(mapped);
  const Header& h = *impl->header;
  if (!std::equal(kMagic.begin(), kMagic.end(), h.magic)) return Error(StatusCode::kCorruptPackage, "bad AIM magic");
  if (h.schema_major != 1 || h.header_size != sizeof(Header) || h.endian_tag != kEndianTag)
    return Error(StatusCode::kCorruptPackage, "unsupported AIM schema, header, or byte order");
  if (h.runtime_abi != kRuntimeAbi) return Error(StatusCode::kIncompatibleAbi, "AIM Runtime ABI mismatch");
  if (h.platform != HostPlatformId())
    return Error(StatusCode::kIncompatiblePlatform, "AIM platform does not match host " + HostPlatform());
  if (h.file_size != impl->size || !RegionValid(impl->size, h.manifest_offset, h.manifest_size) ||
      !RegionValid(impl->size, h.graph_offset, h.graph_size) || !RegionValid(impl->size, h.tensor_offset, h.tensor_size) ||
      !RegionValid(impl->size, h.directory_offset, h.directory_size) || h.directory_size != h.variant_count * sizeof(Variant))
    return Error(StatusCode::kCorruptPackage, "invalid AIM section bounds");
  if (!CheckHash(impl->data + h.directory_offset, h.directory_size, h.directory_sha) ||
      !CheckHash(impl->data + h.manifest_offset, h.manifest_size, h.manifest_sha) ||
      !CheckHash(impl->data + h.graph_offset, h.graph_size, h.graph_sha))
    return Error(StatusCode::kCorruptPackage, "AIM metadata checksum mismatch");
  const auto file_hash = FileHash(impl->data, impl->size);
  if (!HashEquals(file_hash.data(), h.file_sha)) return Error(StatusCode::kCorruptPackage, "AIM file checksum mismatch");
  if (std::any_of(std::begin(h.reserved), std::end(h.reserved), [](std::uint8_t value) { return value != 0; }))
    return Error(StatusCode::kCorruptPackage, "non-zero AIM reserved bytes");
  impl->variants = reinterpret_cast<const Variant*>(impl->data + h.directory_offset);
  for (std::uint32_t i = 0; i < h.variant_count; ++i) {
    const auto& v = impl->variants[i];
    if (!ArchMatchesPlatform(h.platform, v.arch) || v.flags != 0 ||
        std::any_of(std::begin(v.reserved), std::end(v.reserved), [](std::uint8_t value) { return value != 0; }) ||
        v.kernel_offset % 256 != 0 || v.weight_offset % 256 != 0 || v.plan_offset % 256 != 0 ||
        !RegionValid(impl->size, v.kernel_offset, v.kernel_size) || !RegionValid(impl->size, v.weight_offset, v.weight_size) ||
        !RegionValid(impl->size, v.plan_offset, v.plan_size) ||
        !CheckHash(impl->data + v.kernel_offset, v.kernel_size, v.kernel_sha) ||
        !CheckHash(impl->data + v.weight_offset, v.weight_size, v.weight_sha) ||
        !CheckHash(impl->data + v.plan_offset, v.plan_size, v.plan_sha))
      return Error(StatusCode::kCorruptPackage, "invalid variant or payload checksum");
    if (ContainsPtx(impl->data + v.kernel_offset, v.kernel_size))
      return Error(StatusCode::kCorruptPackage, "PTX in an AIM kernel bundle is forbidden");
    if (!IsExactCubin(impl->data + v.kernel_offset, v.kernel_size, v.arch))
      return Error(StatusCode::kCorruptPackage, "kernel bundle is not an exact-architecture NVIDIA CUBIN");
    for (std::uint32_t j = 0; j < i; ++j) if (impl->variants[j].arch == v.arch)
      return Error(StatusCode::kCorruptPackage, "duplicate CUDA architecture variant");
  }
  impl->manifest.assign(reinterpret_cast<const char*>(impl->data + h.manifest_offset), h.manifest_size);
  return Model(std::move(impl));
}

StatusOr<Session> Session::Create(Runtime& runtime, Model& model, const SessionOptions& options) {
  std::uint32_t arch = runtime.impl_->options.cuda_arch_override;
  const Variant* selected = nullptr;
  for (std::uint32_t i = 0; i < model.impl_->header->variant_count; ++i)
    if (model.impl_->variants[i].arch == arch) selected = &model.impl_->variants[i];
  if (selected == nullptr)
    return Error(StatusCode::kIncompatibleArchitecture, "AIM has no exact " + ArchName(arch) + " variant; fallback and PTX JIT are forbidden");
  const auto driver_min = JsonUnsigned(model.impl_->manifest, "cuda_driver_min");
  const auto runtime_min = JsonUnsigned(model.impl_->manifest, "cuda_runtime_min");
  const auto runtime_max = JsonUnsigned(model.impl_->manifest, "cuda_runtime_max");
  const auto cublas = JsonUnsigned(model.impl_->manifest, "cublaslt_abi");
  const auto cudnn = JsonUnsigned(model.impl_->manifest, "cudnn_abi");
  if (!driver_min.ok()) return driver_min.status();
  if (!runtime_min.ok()) return runtime_min.status();
  if (!runtime_max.ok()) return runtime_max.status();
  if (!cublas.ok()) return cublas.status();
  if (!cudnn.ok()) return cudnn.status();
  const auto& ro = runtime.impl_->options;
  if (ro.cuda_driver_version_override < driver_min.value() || ro.cuda_runtime_version_override < runtime_min.value() ||
      ro.cuda_runtime_version_override > runtime_max.value() || ro.cublaslt_abi_override != cublas.value() ||
      ro.cudnn_abi_override != cudnn.value())
    return Error(StatusCode::kIncompatibleAbi, "CUDA/cuBLASLt/cuDNN compatibility check failed");
  auto impl = std::make_unique<Impl>(); impl->runtime = runtime.impl_.get(); impl->model = model.impl_.get();
  impl->variant = selected; impl->profile = options.profile; return Session(std::move(impl));
}

TargetInfo Session::GetTargetInfo() const {
  return {HostPlatform(), ArchName(impl_->variant->arch), impl_->model->header->runtime_abi};
}
std::vector<std::string> Session::GetInputInfo() const { return {"input_ids", "pixel_values", "state"}; }
std::vector<std::string> Session::GetOutputInfo() const { return {"actions"}; }
StatusOr<WorkspaceInfo> Session::GetRequiredWorkspace(const std::string& requested_profile) const {
  const std::string_view plan(reinterpret_cast<const char*>(impl_->model->data + impl_->variant->plan_offset), impl_->variant->plan_size);
  const std::string& profile = requested_profile.empty() ? impl_->profile : requested_profile;
  if (!profile.empty()) {
    const std::string canonical_name = "\"profile\":\"" + profile + "\"";
    if (plan.find(canonical_name) == std::string_view::npos)
      return Error(StatusCode::kInvalidArgument, "profile is not present in the static execution plan: " + profile);
  }
  auto arena = JsonUnsigned(plan, "arena_bytes"); auto workspace = JsonUnsigned(plan, "workspace_bytes");
  if (!arena.ok()) return arena.status();
  if (!workspace.ok()) return workspace.status();
  return WorkspaceInfo{arena.value(), workspace.value()};
}
Status Session::Enqueue(const std::vector<TensorView>&, const std::vector<TensorView>&,
                        const RunOptions&, CudaStream) {
  return Error(StatusCode::kUnimplemented, "kernel launch backend is not included in the schema/runtime milestone");
}

const char* StatusCodeName(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::kOk: return "OK"; case StatusCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case StatusCode::kNotFound: return "NOT_FOUND"; case StatusCode::kIoError: return "IO_ERROR";
    case StatusCode::kCorruptPackage: return "CORRUPT_PACKAGE"; case StatusCode::kIncompatiblePlatform: return "INCOMPATIBLE_PLATFORM";
    case StatusCode::kIncompatibleArchitecture: return "INCOMPATIBLE_ARCHITECTURE"; case StatusCode::kIncompatibleAbi: return "INCOMPATIBLE_ABI";
    case StatusCode::kCudaError: return "CUDA_ERROR"; case StatusCode::kOutOfMemory: return "OUT_OF_MEMORY";
    case StatusCode::kUnimplemented: return "UNIMPLEMENTED";
  }
  return "UNKNOWN";
}
}  // namespace aginfer
