#include "providers/aot_activation.h"

#include "cuda_kernel_payload.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace aginfer::internal {
namespace {

Status Invalid(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}

bool Aligned(const void* pointer, std::uint32_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

bool IsActivation(CudaKernelId kernel_id) noexcept {
  return kernel_id == CudaKernelId::kGeluF32 ||
         kernel_id == CudaKernelId::kGeluBf16 ||
         kernel_id == CudaKernelId::kSiluF32;
}

std::uint32_t DTypeBytes(CudaKernelDType dtype) noexcept {
  return dtype == CudaKernelDType::kF32 ? 4U : 2U;
}

}  // namespace

struct AotActivationCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* input = nullptr;
  void* output = nullptr;
  std::uint64_t numel = 0;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotActivationCommand::AotActivationCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotActivationCommand::~AotActivationCommand() = default;

Status AotActivationCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotActivationBindings& bindings,
    std::unique_ptr<AotActivationCommand>* output) {
  if (output == nullptr) return Invalid("AOT activation command output is null");
  output->reset();
  CudaKernelPayloadView parsed;
  Status status = ParseCudaKernelPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (parsed.launch_abi != CudaKernelLaunchAbi::kUnaryPointersNumel ||
      !IsActivation(parsed.kernel_id)) {
    return Invalid("AOT activation command received a non-activation payload");
  }
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT activation target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT activation module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("AOT activation command needs a loaded caller-owned CUDA module");
  }
  if (bindings.input == nullptr || bindings.output == nullptr ||
      bindings.input == bindings.output ||
      !Aligned(bindings.input, parsed.input_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment)) {
    return Invalid("AOT activation binding is null, misaligned, or aliases output");
  }
  const std::uint64_t tensor_bytes =
      parsed.numel * DTypeBytes(parsed.input_dtype);
  if (bindings.input_bytes < tensor_bytes ||
      bindings.output_bytes < tensor_bytes) {
    return Invalid("AOT activation binding is shorter than the exact problem");
  }
  const char* symbol = CudaKernelSymbol(parsed.kernel_id);
  if (symbol == nullptr) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT activation payload names an unknown numeric kernel ID");
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
  output->reset(new AotActivationCommand(std::move(impl)));
  return Status::Ok();
}

Status AotActivationCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("AOT activation command is not prepared");
  void* arguments[] = {&impl_->input, &impl_->output, &impl_->numel};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
