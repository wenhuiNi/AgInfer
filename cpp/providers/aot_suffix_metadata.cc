#include "providers/aot_suffix_metadata.h"

#include "suffix_metadata_payload.h"

#include <algorithm>
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

struct AotSuffixMetadataCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* prefix_pad = nullptr;
  void* attention_mask = nullptr;
  void* positions = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
};

AotSuffixMetadataCommand::AotSuffixMetadataCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AotSuffixMetadataCommand::~AotSuffixMetadataCommand() = default;

Status AotSuffixMetadataCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotSuffixMetadataBindings& bindings,
    std::unique_ptr<AotSuffixMetadataCommand>* output) {
  if (output == nullptr) return Invalid("suffix metadata command output is null");
  output->reset();
  SuffixMetadataPayloadView parsed;
  Status status = ParseSuffixMetadataPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "suffix metadata target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes || module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "suffix metadata module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("suffix metadata needs a loaded caller-owned CUDA module");
  }
  if (bindings.prefix_pad == nullptr || bindings.attention_mask == nullptr ||
      bindings.positions == nullptr ||
      !Aligned(bindings.prefix_pad, parsed.pad_alignment) ||
      !Aligned(bindings.attention_mask, parsed.mask_alignment) ||
      !Aligned(bindings.positions, parsed.position_alignment)) {
    return Invalid("suffix metadata binding is null or misaligned");
  }
  if (bindings.prefix_pad_bytes < parsed.pad_bytes ||
      bindings.attention_mask_bytes < parsed.mask_bytes ||
      bindings.positions_bytes < parsed.position_bytes) {
    return Invalid("suffix metadata binding is shorter than the exact problem");
  }
  if (Overlaps(bindings.prefix_pad, parsed.pad_bytes, bindings.attention_mask,
               parsed.mask_bytes) ||
      Overlaps(bindings.prefix_pad, parsed.pad_bytes, bindings.positions,
               parsed.position_bytes) ||
      Overlaps(bindings.attention_mask, parsed.mask_bytes, bindings.positions,
               parsed.position_bytes)) {
    return Invalid("suffix metadata buffers overlap");
  }
  auto function = driver->GetFunction(module, SuffixMetadataSymbol());
  if (!function.ok()) return function.status();
  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->prefix_pad = bindings.prefix_pad;
  impl->attention_mask = bindings.attention_mask;
  impl->positions = bindings.positions;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  output->reset(new AotSuffixMetadataCommand(std::move(impl)));
  return Status::Ok();
}

Status AotSuffixMetadataCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("suffix metadata command is not prepared");
  void* arguments[] = {
      &impl_->prefix_pad, &impl_->attention_mask, &impl_->positions};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), 0, cuda_stream, arguments);
}

}  // namespace aginfer::internal
