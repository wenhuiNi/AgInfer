#include "providers/aot_adaptive_rms_norm.h"

#include "adaptive_rms_norm_payload.h"

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

}  // namespace

struct AotAdaptiveRmsNormCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* hidden = nullptr;
  const void* modulation = nullptr;
  void* normalized = nullptr;
  void* gate = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotAdaptiveRmsNormCommand::AotAdaptiveRmsNormCommand(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotAdaptiveRmsNormCommand::~AotAdaptiveRmsNormCommand() = default;

Status AotAdaptiveRmsNormCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotAdaptiveRmsNormBindings& bindings,
    std::unique_ptr<AotAdaptiveRmsNormCommand>* output) {
  if (output == nullptr) {
    return Invalid("AOT adaptive RMSNorm command output is null");
  }
  output->reset();
  AdaptiveRmsNormPayloadView parsed;
  Status status = ParseAdaptiveRmsNormPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT adaptive RMSNorm target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(
        StatusCode::kIncompatibleAbi,
        "AOT adaptive RMSNorm module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid(
        "AOT adaptive RMSNorm needs a loaded caller-owned CUDA module");
  }
  if (bindings.hidden == nullptr || bindings.modulation == nullptr ||
      bindings.normalized == nullptr || bindings.gate == nullptr ||
      !Aligned(bindings.hidden, parsed.hidden_alignment) ||
      !Aligned(bindings.modulation, parsed.modulation_alignment) ||
      !Aligned(bindings.normalized, parsed.normalized_alignment) ||
      !Aligned(bindings.gate, parsed.gate_alignment) ||
      bindings.normalized == bindings.hidden ||
      bindings.normalized == bindings.modulation ||
      bindings.gate == bindings.hidden ||
      bindings.gate == bindings.modulation ||
      bindings.gate == bindings.normalized) {
    return Invalid(
        "AOT adaptive RMSNorm binding is null, misaligned, or aliases output");
  }
  if (bindings.hidden_bytes < parsed.hidden_bytes ||
      bindings.modulation_bytes < parsed.modulation_bytes ||
      bindings.normalized_bytes < parsed.normalized_bytes ||
      bindings.gate_bytes < parsed.gate_bytes) {
    return Invalid(
        "AOT adaptive RMSNorm binding is shorter than the exact problem");
  }
  auto function = driver->GetFunction(module, AdaptiveRmsNormSymbol());
  if (!function.ok()) return function.status();

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->hidden = bindings.hidden;
  impl->modulation = bindings.modulation;
  impl->normalized = bindings.normalized;
  impl->gate = bindings.gate;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  impl->shared_bytes = parsed.shared_bytes;
  output->reset(new AotAdaptiveRmsNormCommand(std::move(impl)));
  return Status::Ok();
}

Status AotAdaptiveRmsNormCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) {
    return Invalid("AOT adaptive RMSNorm command is not prepared");
  }
  void* arguments[] = {&impl_->hidden, &impl_->modulation,
                       &impl_->normalized, &impl_->gate};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
