#include "providers/aot_layer_norm.h"

#include "layer_norm_payload.h"

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

struct AotLayerNormCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* input = nullptr;
  const void* weight = nullptr;
  const void* bias = nullptr;
  void* output = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotLayerNormCommand::AotLayerNormCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotLayerNormCommand::~AotLayerNormCommand() = default;

Status AotLayerNormCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotLayerNormBindings& bindings,
    std::unique_ptr<AotLayerNormCommand>* output) {
  if (output == nullptr) return Invalid("AOT LayerNorm command output is null");
  output->reset();
  LayerNormPayloadView parsed;
  Status status = ParseLayerNormPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT LayerNorm target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT LayerNorm module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("AOT LayerNorm command needs a loaded caller-owned CUDA module");
  }
  if (bindings.input == nullptr || bindings.weight == nullptr ||
      bindings.bias == nullptr || bindings.output == nullptr ||
      !Aligned(bindings.input, parsed.input_alignment) ||
      !Aligned(bindings.weight, parsed.parameter_alignment) ||
      !Aligned(bindings.bias, parsed.parameter_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment) ||
      bindings.output == bindings.input || bindings.output == bindings.weight ||
      bindings.output == bindings.bias) {
    return Invalid("AOT LayerNorm binding is null, misaligned, or aliases output");
  }
  if (bindings.input_bytes < parsed.input_bytes ||
      bindings.weight_bytes < parsed.weight_bytes ||
      bindings.bias_bytes < parsed.bias_bytes ||
      bindings.output_bytes < parsed.output_bytes) {
    return Invalid("AOT LayerNorm binding is shorter than the exact problem");
  }
  auto function = driver->GetFunction(module, kLayerNormSymbol);
  if (!function.ok()) return function.status();

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->input = bindings.input;
  impl->weight = bindings.weight;
  impl->bias = bindings.bias;
  impl->output = bindings.output;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  impl->shared_bytes = parsed.shared_bytes;
  output->reset(new AotLayerNormCommand(std::move(impl)));
  return Status::Ok();
}

Status AotLayerNormCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("AOT LayerNorm command is not prepared");
  void* arguments[] = {
      &impl_->input, &impl_->weight, &impl_->bias, &impl_->output};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
