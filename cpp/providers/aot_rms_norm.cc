#include "providers/aot_rms_norm.h"

#include "rms_norm_payload.h"

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

struct AotRmsNormCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* input = nullptr;
  const void* weight = nullptr;
  void* output = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotRmsNormCommand::AotRmsNormCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotRmsNormCommand::~AotRmsNormCommand() = default;

Status AotRmsNormCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotRmsNormBindings& bindings,
    std::unique_ptr<AotRmsNormCommand>* output) {
  if (output == nullptr) return Invalid("AOT RMSNorm command output is null");
  output->reset();
  RmsNormPayloadView parsed;
  Status status = ParseRmsNormPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT RMSNorm target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT RMSNorm module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("AOT RMSNorm command needs a loaded caller-owned CUDA module");
  }
  if (bindings.input == nullptr || bindings.weight == nullptr ||
      bindings.output == nullptr ||
      !Aligned(bindings.input, parsed.input_alignment) ||
      !Aligned(bindings.weight, parsed.weight_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment) ||
      bindings.output == bindings.input || bindings.output == bindings.weight) {
    return Invalid("AOT RMSNorm binding is null, misaligned, or aliases output");
  }
  if (bindings.input_bytes < parsed.input_bytes ||
      bindings.weight_bytes < parsed.weight_bytes ||
      bindings.output_bytes < parsed.output_bytes) {
    return Invalid("AOT RMSNorm binding is shorter than the exact problem");
  }
  const char* symbol = RmsNormSymbol(parsed.variant);
  if (symbol == nullptr) return Invalid("AOT RMSNorm variant has no symbol");
  auto function = driver->GetFunction(module, symbol);
  if (!function.ok()) return function.status();

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->input = bindings.input;
  impl->weight = bindings.weight;
  impl->output = bindings.output;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  impl->shared_bytes = parsed.shared_bytes;
  output->reset(new AotRmsNormCommand(std::move(impl)));
  return Status::Ok();
}

Status AotRmsNormCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("AOT RMSNorm command is not prepared");
  void* arguments[] = {&impl_->input, &impl_->weight, &impl_->output};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
