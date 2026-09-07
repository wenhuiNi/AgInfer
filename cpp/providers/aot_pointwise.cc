#include "providers/aot_pointwise.h"

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

bool IsPointwise(CudaKernelId kernel_id) noexcept {
  return kernel_id == CudaKernelId::kAddF32 ||
         kernel_id == CudaKernelId::kAddBf16 ||
         kernel_id == CudaKernelId::kAddI32 ||
         kernel_id == CudaKernelId::kMulF32 ||
         kernel_id == CudaKernelId::kMulBf16;
}

}  // namespace

struct AotPointwiseCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* lhs = nullptr;
  const void* rhs = nullptr;
  void* output = nullptr;
  std::uint64_t numel = 0;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotPointwiseCommand::AotPointwiseCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotPointwiseCommand::~AotPointwiseCommand() = default;

Status AotPointwiseCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotPointwiseBindings& bindings,
    std::unique_ptr<AotPointwiseCommand>* output) {
  if (output == nullptr) return Invalid("AOT pointwise command output is null");
  output->reset();
  CudaKernelPayloadView parsed;
  Status status = ParseCudaKernelPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (parsed.launch_abi != CudaKernelLaunchAbi::kBinaryPointersNumel ||
      !IsPointwise(parsed.kernel_id)) {
    return Invalid("AOT pointwise command received a non-binary payload");
  }
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT pointwise target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT pointwise module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("AOT pointwise command needs a loaded caller-owned CUDA module");
  }
  if (bindings.lhs == nullptr || bindings.rhs == nullptr ||
      bindings.output == nullptr ||
      !Aligned(bindings.lhs, parsed.input_alignment) ||
      !Aligned(bindings.rhs, parsed.input_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment)) {
    return Invalid("AOT pointwise tensor binding is null or misaligned");
  }
  const std::uint64_t input_bytes =
      parsed.numel * DTypeBytes(parsed.input_dtype);
  const std::uint64_t output_bytes =
      parsed.numel * DTypeBytes(parsed.output_dtype);
  if (bindings.lhs_bytes < input_bytes || bindings.rhs_bytes < input_bytes ||
      bindings.output_bytes < output_bytes) {
    return Invalid("AOT pointwise tensor binding is shorter than the exact problem");
  }
  const char* symbol = CudaKernelSymbol(parsed.kernel_id);
  auto function = driver->GetFunction(module, symbol);
  if (!function.ok()) return function.status();

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->lhs = bindings.lhs;
  impl->rhs = bindings.rhs;
  impl->output = bindings.output;
  impl->numel = parsed.numel;
  impl->grid = {parsed.grid_x, 1, 1};
  impl->block = {parsed.block_x, 1, 1};
  impl->shared_bytes = parsed.shared_bytes;
  output->reset(new AotPointwiseCommand(std::move(impl)));
  return Status::Ok();
}

Status AotPointwiseCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("AOT pointwise command is not prepared");
  void* arguments[] = {
      &impl_->lhs, &impl_->rhs, &impl_->output, &impl_->numel};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
