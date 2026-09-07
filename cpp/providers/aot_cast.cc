#include "providers/aot_cast.h"

#include "cuda_kernel_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace aginfer::internal {
namespace {

Status Invalid(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}

Status IncompatibleArch(const std::string& message) {
  return Status(StatusCode::kIncompatibleArchitecture, message);
}

Status IncompatibleAbi(const std::string& message) {
  return Status(StatusCode::kIncompatibleAbi, message);
}

bool Aligned(const void* pointer, std::uint32_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

std::uint32_t DTypeBytes(CudaKernelDType dtype) noexcept {
  switch (dtype) {
    case CudaKernelDType::kF32:
    case CudaKernelDType::kI32:
      return 4;
    case CudaKernelDType::kBf16:
      return 2;
    case CudaKernelDType::kBool:
      return 1;
  }
  return 0;
}

}  // namespace

struct AotCastCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* input = nullptr;
  void* output = nullptr;
  std::uint64_t numel = 0;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotCastCommand::AotCastCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotCastCommand::~AotCastCommand() = default;

Status AotCastCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotCastBindings& bindings,
    std::unique_ptr<AotCastCommand>* output) {
  if (output == nullptr) return Invalid("AOT cast command output is null");
  output->reset();
  CudaKernelPayloadView parsed;
  Status status = ParseCudaKernelPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (parsed.launch_abi != CudaKernelLaunchAbi::kUnaryPointersNumel ||
      (parsed.kernel_id != CudaKernelId::kCastBf16ToF32 &&
       parsed.kernel_id != CudaKernelId::kCastF32ToBf16 &&
       parsed.kernel_id != CudaKernelId::kCastBoolToI32)) {
    return Invalid("AOT cast command received a non-cast kernel payload");
  }
  if (active_arch != parsed.target_arch) {
    return IncompatibleArch("AOT cast target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return IncompatibleAbi("AOT cast module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("AOT cast command needs a loaded caller-owned CUDA module");
  }
  if (bindings.input == nullptr || bindings.output == nullptr ||
      !Aligned(bindings.input, parsed.input_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment)) {
    return Invalid("AOT cast tensor binding is null or misaligned");
  }
  const std::uint64_t input_bytes =
      parsed.numel * DTypeBytes(parsed.input_dtype);
  const std::uint64_t output_bytes =
      parsed.numel * DTypeBytes(parsed.output_dtype);
  if (bindings.input_bytes < input_bytes ||
      bindings.output_bytes < output_bytes) {
    return Invalid("AOT cast tensor binding is shorter than the exact problem");
  }
  const char* symbol = CudaKernelSymbol(parsed.kernel_id);
  if (symbol == nullptr) {
    return IncompatibleAbi("AOT cast payload names an unknown numeric kernel ID");
  }
  auto function = driver->GetFunction(module, symbol);
  if (!function.ok()) return function.status();

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->input = bindings.input;
  impl->output = bindings.output;
  impl->numel = parsed.numel;
  impl->grid = {parsed.grid_x, 1, 1};
  impl->block = {parsed.block_x, 1, 1};
  impl->shared_bytes = parsed.shared_bytes;
  output->reset(new AotCastCommand(std::move(impl)));
  return Status::Ok();
}

Status AotCastCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("AOT cast command is not prepared");
  void* arguments[] = {&impl_->input, &impl_->output, &impl_->numel};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
