#include "providers/aot_action_slice.h"

#include "action_slice_payload.h"

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
bool Aligned(const void* p, std::uint32_t a) {
  return reinterpret_cast<std::uintptr_t>(p) % a == 0;
}
bool Overlaps(const void* a, std::uint64_t an, const void* b,
              std::uint64_t bn) {
  const auto av = reinterpret_cast<std::uintptr_t>(a);
  const auto bv = reinterpret_cast<std::uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}

}  // namespace

struct AotActionSliceCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* input = nullptr;
  void* output = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
};

AotActionSliceCommand::AotActionSliceCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AotActionSliceCommand::~AotActionSliceCommand() = default;

Status AotActionSliceCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotActionSliceBindings& bindings,
    std::unique_ptr<AotActionSliceCommand>* output) {
  if (output == nullptr) return Invalid("action slice command output is null");
  output->reset();
  ActionSlicePayloadView parsed;
  Status status = ParseActionSlicePayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "action slice target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes || module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "action slice module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("action slice needs a loaded caller-owned CUDA module");
  }
  if (bindings.input == nullptr || bindings.output == nullptr ||
      !Aligned(bindings.input, parsed.input_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment)) {
    return Invalid("action slice binding is null or misaligned");
  }
  if (bindings.input_bytes < parsed.input_bytes ||
      bindings.output_bytes < parsed.output_bytes) {
    return Invalid("action slice binding is shorter than the exact problem");
  }
  if (Overlaps(bindings.input, parsed.input_bytes, bindings.output,
               parsed.output_bytes)) {
    return Invalid("action slice input and output overlap");
  }
  auto function = driver->GetFunction(module, ActionSliceSymbol());
  if (!function.ok()) return function.status();
  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->input = bindings.input;
  impl->output = bindings.output;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  output->reset(new AotActionSliceCommand(std::move(impl)));
  return Status::Ok();
}

Status AotActionSliceCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("action slice command is not prepared");
  void* arguments[] = {&impl_->input, &impl_->output};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), 0, cuda_stream, arguments);
}

}  // namespace aginfer::internal
