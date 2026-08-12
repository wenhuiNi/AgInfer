#include "cuda_driver.h"

#include <dlfcn.h>

#include <limits>
#include <string>
#include <utility>

namespace aginfer::internal {
namespace {

using CuResult = int;
using CuDevice = int;
using CuContext = void*;
using CuStream = void*;
constexpr CuResult kCudaSuccess = 0;

template <typename T>
T Symbol(void* library, const char* name) {
  return reinterpret_cast<T>(dlsym(library, name));
}

}  // namespace

struct CudaDriver::Impl {
  void* library = nullptr;
  CuDevice device = 0;
  CuContext context = nullptr;
  bool retained_primary = false;

  CuResult (*get_error_name)(CuResult, const char**) = nullptr;
  CuResult (*get_error_string)(CuResult, const char**) = nullptr;
  CuResult (*ctx_set_current)(CuContext) = nullptr;
  CuResult (*primary_release)(CuDevice) = nullptr;
  CuResult (*module_load_data)(CuModule*, const void*) = nullptr;
  CuResult (*module_get_function)(CuFunction*, CuModule, const char*) = nullptr;
  CuResult (*module_unload)(CuModule) = nullptr;
  CuResult (*mem_alloc)(CuDevicePtr*, std::size_t) = nullptr;
  CuResult (*mem_free)(CuDevicePtr) = nullptr;
  CuResult (*memcpy_htod)(CuDevicePtr, const void*, std::size_t) = nullptr;
  CuResult (*launch_kernel)(CuFunction, unsigned, unsigned, unsigned,
                            unsigned, unsigned, unsigned, unsigned,
                            CuStream, void**, void**) = nullptr;

  std::string ErrorText(CuResult result, const std::string& operation) const {
    const char* name = nullptr;
    const char* description = nullptr;
    if (get_error_name != nullptr) get_error_name(result, &name);
    if (get_error_string != nullptr) get_error_string(result, &description);
    std::string message = operation + " failed";
    if (name != nullptr) message += ": " + std::string(name);
    if (description != nullptr) message += " (" + std::string(description) + ")";
    return message;
  }

  ~Impl() {
    if (context != nullptr && ctx_set_current != nullptr) ctx_set_current(context);
    if (retained_primary && primary_release != nullptr) primary_release(device);
    if (library != nullptr) dlclose(library);
  }
};

CudaDriver::CudaDriver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaDriver::CudaDriver(CudaDriver&&) noexcept = default;
CudaDriver& CudaDriver::operator=(CudaDriver&&) noexcept = default;
CudaDriver::~CudaDriver() = default;

StatusOr<CudaDriver> CudaDriver::Create(std::uint32_t expected_arch) {
  auto impl = std::make_unique<Impl>();
  impl->library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (impl->library == nullptr) impl->library = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
  if (impl->library == nullptr)
    return Status(StatusCode::kCudaError, "cannot load the CUDA Driver library");

  using Init = CuResult (*)(unsigned);
  using DeviceGet = CuResult (*)(CuDevice*, int);
  using AttributeGet = CuResult (*)(int*, int, CuDevice);
  using ContextGet = CuResult (*)(CuContext*);
  using PrimaryRetain = CuResult (*)(CuContext*, CuDevice);
  auto init = Symbol<Init>(impl->library, "cuInit");
  auto device_get = Symbol<DeviceGet>(impl->library, "cuDeviceGet");
  auto attribute_get = Symbol<AttributeGet>(impl->library, "cuDeviceGetAttribute");
  auto context_get = Symbol<ContextGet>(impl->library, "cuCtxGetCurrent");
  auto primary_retain = Symbol<PrimaryRetain>(impl->library, "cuDevicePrimaryCtxRetain");
  impl->ctx_set_current = Symbol<decltype(impl->ctx_set_current)>(impl->library, "cuCtxSetCurrent");
  impl->primary_release = Symbol<decltype(impl->primary_release)>(impl->library, "cuDevicePrimaryCtxRelease_v2");
  if (impl->primary_release == nullptr)
    impl->primary_release = Symbol<decltype(impl->primary_release)>(impl->library, "cuDevicePrimaryCtxRelease");
  impl->get_error_name = Symbol<decltype(impl->get_error_name)>(impl->library, "cuGetErrorName");
  impl->get_error_string = Symbol<decltype(impl->get_error_string)>(impl->library, "cuGetErrorString");
  impl->module_load_data = Symbol<decltype(impl->module_load_data)>(impl->library, "cuModuleLoadData");
  impl->module_get_function = Symbol<decltype(impl->module_get_function)>(impl->library, "cuModuleGetFunction");
  impl->module_unload = Symbol<decltype(impl->module_unload)>(impl->library, "cuModuleUnload");
  impl->mem_alloc = Symbol<decltype(impl->mem_alloc)>(impl->library, "cuMemAlloc_v2");
  impl->mem_free = Symbol<decltype(impl->mem_free)>(impl->library, "cuMemFree_v2");
  impl->memcpy_htod = Symbol<decltype(impl->memcpy_htod)>(impl->library, "cuMemcpyHtoD_v2");
  impl->launch_kernel = Symbol<decltype(impl->launch_kernel)>(impl->library, "cuLaunchKernel");
  if (init == nullptr || device_get == nullptr || attribute_get == nullptr || context_get == nullptr ||
      primary_retain == nullptr || impl->ctx_set_current == nullptr || impl->primary_release == nullptr ||
      impl->module_load_data == nullptr || impl->module_get_function == nullptr || impl->module_unload == nullptr ||
      impl->mem_alloc == nullptr || impl->mem_free == nullptr || impl->memcpy_htod == nullptr ||
      impl->launch_kernel == nullptr)
    return Status(StatusCode::kCudaError, "CUDA Driver library is missing required symbols");

  CuResult result = init(0);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl->ErrorText(result, "cuInit"));
  result = device_get(&impl->device, 0);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl->ErrorText(result, "cuDeviceGet"));
  int major = 0;
  int minor = 0;
  result = attribute_get(&major, 75, impl->device);
  if (result == kCudaSuccess) result = attribute_get(&minor, 76, impl->device);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl->ErrorText(result, "cuDeviceGetAttribute"));
  if (static_cast<std::uint32_t>(major * 10 + minor) != expected_arch)
    return Status(StatusCode::kIncompatibleArchitecture,
                  "active CUDA device does not match the selected AIM architecture");
  result = context_get(&impl->context);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl->ErrorText(result, "cuCtxGetCurrent"));
  if (impl->context == nullptr) {
    result = primary_retain(&impl->context, impl->device);
    if (result != kCudaSuccess)
      return Status(StatusCode::kCudaError, impl->ErrorText(result, "cuDevicePrimaryCtxRetain"));
    impl->retained_primary = true;
    result = impl->ctx_set_current(impl->context);
    if (result != kCudaSuccess)
      return Status(StatusCode::kCudaError, impl->ErrorText(result, "cuCtxSetCurrent"));
  }
  return CudaDriver(std::move(impl));
}

Status CudaDriver::MakeCurrent() {
  const CuResult result = impl_->ctx_set_current(impl_->context);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuCtxSetCurrent"));
  return Status::Ok();
}

StatusOr<CuModule> CudaDriver::LoadModule(const void* image) {
  CuModule module = nullptr;
  const CuResult result = impl_->module_load_data(&module, image);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuModuleLoadData"));
  return module;
}

StatusOr<CuFunction> CudaDriver::GetFunction(CuModule module, const std::string& name) {
  CuFunction function = nullptr;
  const CuResult result = impl_->module_get_function(&function, module, name.c_str());
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuModuleGetFunction(" + name + ")"));
  return function;
}

Status CudaDriver::UnloadModule(CuModule module) {
  if (module == nullptr) return Status::Ok();
  const CuResult result = impl_->module_unload(module);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuModuleUnload"));
  return Status::Ok();
}

StatusOr<CuDevicePtr> CudaDriver::Allocate(std::uint64_t bytes) {
  if (bytes == 0) return CuDevicePtr{0};
  if (bytes > std::numeric_limits<std::size_t>::max())
    return Status(StatusCode::kOutOfMemory, "CUDA allocation size exceeds the host size type");
  CuDevicePtr pointer = 0;
  const CuResult result = impl_->mem_alloc(&pointer, static_cast<std::size_t>(bytes));
  if (result != kCudaSuccess)
    return Status(StatusCode::kOutOfMemory, impl_->ErrorText(result, "cuMemAlloc"));
  return pointer;
}

Status CudaDriver::Free(CuDevicePtr pointer) {
  if (pointer == 0) return Status::Ok();
  const CuResult result = impl_->mem_free(pointer);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuMemFree"));
  return Status::Ok();
}

Status CudaDriver::CopyHostToDevice(CuDevicePtr destination, const void* source, std::size_t bytes) {
  if (bytes == 0) return Status::Ok();
  const CuResult result = impl_->memcpy_htod(destination, source, bytes);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuMemcpyHtoD"));
  return Status::Ok();
}

Status CudaDriver::Launch(CuFunction function, const std::uint32_t grid[3],
                          const std::uint32_t block[3], std::uint32_t shared_bytes,
                          CudaStream stream, void** arguments) {
  const CuResult result = impl_->launch_kernel(
      function, grid[0], grid[1], grid[2], block[0], block[1], block[2],
      shared_bytes, reinterpret_cast<CuStream>(stream), arguments, nullptr);
  if (result != kCudaSuccess)
    return Status(StatusCode::kCudaError, impl_->ErrorText(result, "cuLaunchKernel"));
  return Status::Ok();
}

}  // namespace aginfer::internal
