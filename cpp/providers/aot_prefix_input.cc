#include "providers/aot_prefix_input.h"

#include "prefix_input_payload.h"

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

struct AotPrefixInputCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  AotPrefixInputBindings bindings;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
};

AotPrefixInputCommand::AotPrefixInputCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AotPrefixInputCommand::~AotPrefixInputCommand() = default;

Status AotPrefixInputCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotPrefixInputBindings& bindings,
    std::unique_ptr<AotPrefixInputCommand>* output) {
  if (output == nullptr) return Invalid("prefix input command output is null");
  output->reset();
  PrefixInputPayloadView parsed;
  Status status = ParsePrefixInputPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "prefix input target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes || module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "prefix input module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("prefix input needs a loaded caller-owned CUDA module");
  }
  for (std::size_t index = 0; index < 3; ++index) {
    if (bindings.images[index] == nullptr ||
        bindings.image_masks[index] == nullptr ||
        !Aligned(bindings.images[index], parsed.image_alignment) ||
        !Aligned(bindings.image_masks[index], parsed.mask_alignment) ||
        bindings.image_bytes[index] < parsed.image_bytes ||
        bindings.image_mask_bytes[index] < 1) {
      return Invalid("prefix input image binding is null, short, or misaligned");
    }
  }
  if (bindings.embedding == nullptr || bindings.tokens == nullptr ||
      bindings.token_mask == nullptr || bindings.prefix == nullptr ||
      bindings.pad_mask == nullptr || bindings.positions == nullptr ||
      !Aligned(bindings.embedding, parsed.embedding_alignment) ||
      !Aligned(bindings.tokens, parsed.token_alignment) ||
      !Aligned(bindings.token_mask, parsed.mask_alignment) ||
      !Aligned(bindings.prefix, parsed.prefix_alignment) ||
      !Aligned(bindings.pad_mask, parsed.mask_alignment) ||
      !Aligned(bindings.positions, parsed.position_alignment)) {
    return Invalid("prefix input binding is null or misaligned");
  }
  if (bindings.embedding_bytes < parsed.embedding_bytes ||
      bindings.token_bytes < parsed.token_bytes ||
      bindings.token_mask_bytes < 200 ||
      bindings.prefix_bytes < parsed.prefix_bytes ||
      bindings.pad_mask_bytes < 968 || bindings.position_bytes < 3872) {
    return Invalid("prefix input binding is shorter than the exact problem");
  }
  const std::array<const void*, 9> reads{
      bindings.images[0], bindings.images[1], bindings.images[2],
      bindings.embedding, bindings.tokens, bindings.image_masks[0],
      bindings.image_masks[1], bindings.image_masks[2], bindings.token_mask};
  const std::array<std::uint64_t, 9> read_bytes{
      parsed.image_bytes, parsed.image_bytes, parsed.image_bytes,
      parsed.embedding_bytes, parsed.token_bytes, 1, 1, 1, 200};
  const std::array<void*, 3> writes{
      bindings.prefix, bindings.pad_mask, bindings.positions};
  const std::array<std::uint64_t, 3> write_bytes{
      parsed.prefix_bytes, 968, 3872};
  for (std::size_t write = 0; write < writes.size(); ++write) {
    for (std::size_t read = 0; read < reads.size(); ++read) {
      if (Overlaps(writes[write], write_bytes[write], reads[read], read_bytes[read])) {
        return Invalid("prefix input read/write buffers overlap");
      }
    }
    for (std::size_t other = write + 1; other < writes.size(); ++other) {
      if (Overlaps(writes[write], write_bytes[write], writes[other],
                   write_bytes[other])) {
        return Invalid("prefix input output buffers overlap");
      }
    }
  }
  auto function = driver->GetFunction(module, PrefixInputSymbol());
  if (!function.ok()) return function.status();
  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->bindings = bindings;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  output->reset(new AotPrefixInputCommand(std::move(impl)));
  return Status::Ok();
}

Status AotPrefixInputCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("prefix input command is not prepared");
  void* image0 = const_cast<void*>(impl_->bindings.images[0]);
  void* image1 = const_cast<void*>(impl_->bindings.images[1]);
  void* image2 = const_cast<void*>(impl_->bindings.images[2]);
  void* embedding = const_cast<void*>(impl_->bindings.embedding);
  void* tokens = const_cast<void*>(impl_->bindings.tokens);
  void* image_mask0 = const_cast<void*>(impl_->bindings.image_masks[0]);
  void* image_mask1 = const_cast<void*>(impl_->bindings.image_masks[1]);
  void* image_mask2 = const_cast<void*>(impl_->bindings.image_masks[2]);
  void* token_mask = const_cast<void*>(impl_->bindings.token_mask);
  void* prefix = impl_->bindings.prefix;
  void* pad_mask = impl_->bindings.pad_mask;
  void* positions = impl_->bindings.positions;
  void* arguments[] = {
      &image0, &image1, &image2, &embedding, &tokens, &image_mask0,
      &image_mask1, &image_mask2, &token_mask, &prefix, &pad_mask, &positions};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), 0, cuda_stream, arguments);
}

}  // namespace aginfer::internal
