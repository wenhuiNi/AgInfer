#include "providers/aot_vision_attention.h"

#include "vision_attention_payload.h"

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

bool HasOutputAlias(const AotVisionAttentionBindings& bindings) noexcept {
  return bindings.output == bindings.query || bindings.output == bindings.key ||
         bindings.output == bindings.value || bindings.output == bindings.mask;
}

}  // namespace

struct AotVisionAttentionCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* query = nullptr;
  const void* key = nullptr;
  const void* value = nullptr;
  const void* mask = nullptr;
  void* output = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::uint32_t shared_bytes = 0;
};

AotVisionAttentionCommand::AotVisionAttentionCommand(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotVisionAttentionCommand::~AotVisionAttentionCommand() = default;

Status AotVisionAttentionCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotVisionAttentionBindings& bindings,
    std::unique_ptr<AotVisionAttentionCommand>* output) {
  if (output == nullptr) {
    return Invalid("AOT vision attention command output is null");
  }
  output->reset();
  VisionAttentionPayloadView parsed;
  Status status = ParseVisionAttentionPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT vision attention target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT vision attention module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid(
        "AOT vision attention command needs a loaded caller-owned CUDA module");
  }
  if (bindings.query == nullptr || bindings.key == nullptr ||
      bindings.value == nullptr || bindings.mask == nullptr ||
      bindings.output == nullptr || !Aligned(bindings.query, 16) ||
      !Aligned(bindings.key, 16) || !Aligned(bindings.value, 16) ||
      !Aligned(bindings.output, 16) || HasOutputAlias(bindings)) {
    return Invalid(
        "AOT vision attention binding is null, misaligned, or aliases output");
  }
  if (bindings.query_bytes < parsed.query_bytes ||
      bindings.key_bytes < parsed.key_bytes ||
      bindings.value_bytes < parsed.value_bytes ||
      bindings.mask_bytes < parsed.mask_bytes ||
      bindings.output_bytes < parsed.output_bytes) {
    return Invalid(
        "AOT vision attention binding is shorter than the exact problem");
  }
  auto function = driver->GetFunction(module, kVisionAttentionSymbol);
  if (!function.ok()) return function.status();

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->query = bindings.query;
  impl->key = bindings.key;
  impl->value = bindings.value;
  impl->mask = bindings.mask;
  impl->output = bindings.output;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  impl->shared_bytes = parsed.shared_bytes;
  output->reset(new AotVisionAttentionCommand(std::move(impl)));
  return Status::Ok();
}

Status AotVisionAttentionCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) {
    return Invalid("AOT vision attention command is not prepared");
  }
  void* arguments[] = {&impl_->query, &impl_->key, &impl_->value,
                       &impl_->mask, &impl_->output};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), impl_->shared_bytes,
                               cuda_stream, arguments);
}

}  // namespace aginfer::internal
