#include "aginfer/runtime.h"

#include "cuda_driver.h"
#include "plan.h"
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

DType ToPublicDType(std::uint32_t value) {
  switch (static_cast<internal::PlanDType>(value)) {
    case internal::PlanDType::kFp32: return DType::kFp32;
    case internal::PlanDType::kFp16: return DType::kFp16;
    case internal::PlanDType::kBf16: return DType::kBf16;
    case internal::PlanDType::kFp8E4M3: return DType::kFp8E4M3;
    case internal::PlanDType::kInt32: return DType::kInt32;
  }
  return DType::kFp32;
}

MemoryLocation ToPublicLocation(std::uint32_t value) {
  return value == static_cast<std::uint32_t>(internal::PlanLocation::kDevice)
             ? MemoryLocation::kDevice
             : MemoryLocation::kHost;
}

TensorInfo ToTensorInfo(const internal::ParsedPlan& plan,
                        const internal::PlanTensor& tensor) {
  TensorInfo result;
  result.name = std::string(plan.String(tensor.name_offset));
  result.dtype = ToPublicDType(tensor.dtype);
  result.shape.assign(tensor.shape, tensor.shape + tensor.rank);
  result.stride.assign(tensor.stride, tensor.stride + tensor.rank);
  result.byte_size = static_cast<std::size_t>(tensor.byte_size);
  result.location = ToPublicLocation(tensor.location);
  return result;
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
  internal::ParsedPlan plan;
  const internal::PlanProfile* profile = nullptr;
  std::unique_ptr<internal::CudaDriver> cuda;
  internal::CuModule module = nullptr;
  internal::CuDevicePtr weights = 0;
  internal::CuDevicePtr arena = 0;
  internal::CuDevicePtr workspace = 0;
  std::vector<internal::CuFunction> functions;
  std::vector<const TensorView*> tensor_bindings;
  std::vector<std::uint64_t> argument_values;
  std::vector<void*> kernel_parameters;

  Status InitializeCuda();
  Status BindTensors(const std::vector<TensorView>& inputs,
                     const std::vector<TensorView>& outputs);
  Status Launch(CudaStream stream);

  ~Impl() {
    if (cuda == nullptr) return;
    cuda->MakeCurrent();
    cuda->Free(workspace);
    cuda->Free(arena);
    cuda->Free(weights);
    cuda->UnloadModule(module);
  }
};

Status Session::Impl::InitializeCuda() {
  if (cuda != nullptr) return cuda->MakeCurrent();
  auto created = internal::CudaDriver::Create(variant->arch);
  if (!created.ok()) return created.status();
  cuda = std::make_unique<internal::CudaDriver>(std::move(created).value());
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
    const internal::PlanLaunch& launch = plan.launches[profile->first_launch + index];
    auto function = cuda->GetFunction(module, std::string(plan.String(launch.kernel_name_offset)));
    if (!function.ok()) return function.status();
    functions.push_back(function.value());
  }
  return Status::Ok();
}

Status Session::Impl::BindTensors(const std::vector<TensorView>& inputs,
                                  const std::vector<TensorView>& outputs) {
  std::fill(tensor_bindings.begin(), tensor_bindings.end(), nullptr);
  std::size_t expected_inputs = 0;
  std::size_t expected_outputs = 0;
  for (std::uint32_t index = 0; index < profile->tensor_count; ++index) {
    const internal::PlanTensor& tensor = plan.tensors[profile->first_tensor + index];
    if (tensor.io_kind == static_cast<std::uint32_t>(internal::PlanIoKind::kInput)) ++expected_inputs;
    else ++expected_outputs;
  }
  if (inputs.size() != expected_inputs || outputs.size() != expected_outputs)
    return Error(StatusCode::kInvalidArgument, "input or output tensor count does not match the selected profile");

  auto bind_group = [&](const std::vector<TensorView>& views, internal::PlanIoKind kind) -> Status {
    for (const TensorView& view : views) {
      const internal::PlanTensor* expected = nullptr;
      std::uint32_t expected_index = 0;
      for (std::uint32_t index = 0; index < profile->tensor_count; ++index) {
        const std::uint32_t global_index = profile->first_tensor + index;
        const internal::PlanTensor& candidate = plan.tensors[global_index];
        if (candidate.io_kind == static_cast<std::uint32_t>(kind) &&
            plan.String(candidate.name_offset) == view.name) {
          expected = &candidate;
          expected_index = global_index;
          break;
        }
      }
      if (expected == nullptr)
        return Error(StatusCode::kInvalidArgument, "unexpected tensor for selected profile: " + view.name);
      if (tensor_bindings[expected_index] != nullptr)
        return Error(StatusCode::kInvalidArgument, "duplicate TensorView: " + view.name);
      const DType expected_dtype = ToPublicDType(expected->dtype);
      const MemoryLocation expected_location = ToPublicLocation(expected->location);
      if (view.dtype != expected_dtype || view.location != expected_location)
        return Error(StatusCode::kInvalidArgument, "dtype or memory location mismatch for tensor: " + view.name);
      if (view.data == nullptr || view.byte_size < expected->byte_size || view.shape.size() != expected->rank ||
          view.stride.size() != expected->rank)
        return Error(StatusCode::kInvalidArgument, "buffer, byte size, or rank mismatch for tensor: " + view.name);
      for (std::uint32_t dimension = 0; dimension < expected->rank; ++dimension) {
        if (view.shape[dimension] != expected->shape[dimension] ||
            view.stride[dimension] != expected->stride[dimension])
          return Error(StatusCode::kInvalidArgument, "shape or stride is outside the selected profile: " + view.name);
      }
      tensor_bindings[expected_index] = &view;
    }
    return Status::Ok();
  };
  Status status = bind_group(inputs, internal::PlanIoKind::kInput);
  if (!status.ok()) return status;
  status = bind_group(outputs, internal::PlanIoKind::kOutput);
  if (!status.ok()) return status;
  return Status::Ok();
}

Status Session::Impl::Launch(CudaStream stream) {
  for (std::uint32_t launch_index = 0; launch_index < profile->launch_count; ++launch_index) {
    const internal::PlanLaunch& launch = plan.launches[profile->first_launch + launch_index];
    for (std::uint32_t index = 0; index < launch.argument_count; ++index) {
      const internal::PlanArgument& argument = plan.arguments[launch.first_argument + index];
      switch (static_cast<internal::PlanArgKind>(argument.kind)) {
        case internal::PlanArgKind::kTensor:
          argument_values[index] = static_cast<std::uint64_t>(
              reinterpret_cast<std::uintptr_t>(tensor_bindings[argument.index]->data));
          break;
        case internal::PlanArgKind::kScalarU32:
        case internal::PlanArgKind::kScalarF32: argument_values[index] = argument.value; break;
        case internal::PlanArgKind::kArenaOffset: argument_values[index] = arena + argument.offset; break;
        case internal::PlanArgKind::kWeightsOffset: argument_values[index] = weights + argument.offset; break;
      }
      kernel_parameters[index] = &argument_values[index];
    }
    Status status = cuda->Launch(functions[launch_index], launch.grid, launch.block,
                                 launch.shared_bytes, stream, kernel_parameters.data());
    if (!status.ok()) return status;
  }
  return Status::Ok();
}

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
  auto impl = std::make_unique<Impl>();
  impl->runtime = runtime.impl_.get();
  impl->model = model.impl_.get();
  impl->variant = selected;
  Status plan_status = internal::ParsePlan(
      model.impl_->data + selected->plan_offset,
      static_cast<std::size_t>(selected->plan_size), selected->arch,
      selected->weight_size, &impl->plan);
  if (!plan_status.ok()) return plan_status;
  if (options.profile.empty()) impl->profile = &impl->plan.profiles[0];
  else impl->profile = impl->plan.FindProfile(options.profile);
  if (impl->profile == nullptr)
    return Error(StatusCode::kInvalidArgument, "profile is not present in the static execution plan: " + options.profile);
  impl->tensor_bindings.resize(impl->plan.header->tensor_count);
  std::uint32_t max_arguments = 0;
  for (std::uint32_t index = 0; index < impl->profile->launch_count; ++index) {
    max_arguments = std::max(max_arguments,
        impl->plan.launches[impl->profile->first_launch + index].argument_count);
  }
  impl->argument_values.resize(max_arguments);
  impl->kernel_parameters.resize(max_arguments);
  return Session(std::move(impl));
}

TargetInfo Session::GetTargetInfo() const {
  return {HostPlatform(), ArchName(impl_->variant->arch), impl_->model->header->runtime_abi};
}
std::vector<TensorInfo> Session::GetInputInfo() const {
  std::vector<TensorInfo> result;
  for (std::uint32_t index = 0; index < impl_->profile->tensor_count; ++index) {
    const internal::PlanTensor& tensor = impl_->plan.tensors[impl_->profile->first_tensor + index];
    if (tensor.io_kind == static_cast<std::uint32_t>(internal::PlanIoKind::kInput))
      result.emplace_back(ToTensorInfo(impl_->plan, tensor));
  }
  return result;
}
std::vector<TensorInfo> Session::GetOutputInfo() const {
  std::vector<TensorInfo> result;
  for (std::uint32_t index = 0; index < impl_->profile->tensor_count; ++index) {
    const internal::PlanTensor& tensor = impl_->plan.tensors[impl_->profile->first_tensor + index];
    if (tensor.io_kind == static_cast<std::uint32_t>(internal::PlanIoKind::kOutput))
      result.emplace_back(ToTensorInfo(impl_->plan, tensor));
  }
  return result;
}
StatusOr<WorkspaceInfo> Session::GetRequiredWorkspace(const std::string& requested_profile) const {
  if (!requested_profile.empty() && impl_->plan.FindProfile(requested_profile) == nullptr)
    return Error(StatusCode::kInvalidArgument, "profile is not present in the static execution plan: " + requested_profile);
  return WorkspaceInfo{impl_->plan.header->arena_bytes, impl_->plan.header->workspace_bytes};
}
Status Session::Enqueue(const std::vector<TensorView>& inputs, const std::vector<TensorView>& outputs,
                        const RunOptions&, CudaStream stream) {
  Status status = impl_->BindTensors(inputs, outputs);
  if (!status.ok()) return status;
  status = impl_->InitializeCuda();
  if (!status.ok()) return status;
  return impl_->Launch(stream);
}

const char* StatusCodeName(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::kOk: return "OK"; case StatusCode::kInvalidArgument: return "INVALID_ARGUMENT";
    case StatusCode::kNotFound: return "NOT_FOUND"; case StatusCode::kIoError: return "IO_ERROR";
    case StatusCode::kCorruptPackage: return "CORRUPT_PACKAGE"; case StatusCode::kIncompatiblePlatform: return "INCOMPATIBLE_PLATFORM";
    case StatusCode::kIncompatibleArchitecture: return "INCOMPATIBLE_ARCHITECTURE"; case StatusCode::kIncompatibleAbi: return "INCOMPATIBLE_ABI";
    case StatusCode::kCudaError: return "CUDA_ERROR"; case StatusCode::kOutOfMemory: return "OUT_OF_MEMORY";
  }
  return "UNKNOWN";
}
}  // namespace aginfer
