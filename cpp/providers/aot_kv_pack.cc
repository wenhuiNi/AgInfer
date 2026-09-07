#include "providers/aot_kv_pack.h"

#include "kv_pack_payload.h"

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

bool Aligned(const void* pointer, std::uint32_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

bool Overlaps(const void* left, std::uint64_t left_bytes, const void* right,
              std::uint64_t right_bytes) noexcept {
  const auto left_address = reinterpret_cast<std::uintptr_t>(left);
  const auto right_address = reinterpret_cast<std::uintptr_t>(right);
  return left_address <= right_address
             ? right_address - left_address < left_bytes
             : left_address - right_address < right_bytes;
}

}  // namespace

struct AotKvPackCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* prefix_k = nullptr;
  const void* prefix_v = nullptr;
  const void* current_k = nullptr;
  const void* current_v = nullptr;
  void* packed_k = nullptr;
  void* packed_v = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
};

AotKvPackCommand::AotKvPackCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotKvPackCommand::~AotKvPackCommand() = default;

Status AotKvPackCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotKvPackBindings& bindings,
    std::unique_ptr<AotKvPackCommand>* output) {
  if (output == nullptr) return Invalid("AOT KV pack command output is null");
  output->reset();
  KvPackPayloadView parsed;
  Status status = ParseKvPackPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "AOT KV pack target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "AOT KV pack module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("AOT KV pack needs a loaded caller-owned CUDA module");
  }
  const std::array<const void*, 6> pointers{
      bindings.prefix_k, bindings.prefix_v, bindings.current_k,
      bindings.current_v, bindings.packed_k, bindings.packed_v};
  if (std::any_of(pointers.begin(), pointers.end(), [](const void* pointer) {
        return pointer == nullptr;
      }) ||
      std::any_of(pointers.begin(), pointers.end(), [&](const void* pointer) {
        return !Aligned(pointer, parsed.alignment);
      })) {
    return Invalid("AOT KV pack binding is null or misaligned");
  }
  if (bindings.prefix_k_bytes < parsed.prefix_bytes ||
      bindings.prefix_v_bytes < parsed.prefix_bytes ||
      bindings.current_k_bytes < parsed.current_k_bytes ||
      bindings.current_v_bytes < parsed.current_v_bytes ||
      bindings.packed_k_bytes < parsed.packed_k_bytes ||
      bindings.packed_v_bytes < parsed.packed_v_bytes) {
    return Invalid("AOT KV pack binding is shorter than the exact problem");
  }
  const std::array<std::pair<const void*, std::uint64_t>, 4> inputs{{
      {bindings.prefix_k, parsed.prefix_bytes},
      {bindings.prefix_v, parsed.prefix_bytes},
      {bindings.current_k, parsed.current_k_bytes},
      {bindings.current_v, parsed.current_v_bytes},
  }};
  if (Overlaps(bindings.packed_k, parsed.packed_k_bytes, bindings.packed_v,
               parsed.packed_v_bytes) ||
      std::any_of(inputs.begin(), inputs.end(), [&](const auto& input) {
        return Overlaps(input.first, input.second, bindings.packed_k,
                        parsed.packed_k_bytes) ||
               Overlaps(input.first, input.second, bindings.packed_v,
                        parsed.packed_v_bytes);
      })) {
    return Invalid("AOT KV pack output overlaps an input or another output");
  }
  auto function = driver->GetFunction(module, KvPackSymbol());
  if (!function.ok()) return function.status();
  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->prefix_k = bindings.prefix_k;
  impl->prefix_v = bindings.prefix_v;
  impl->current_k = bindings.current_k;
  impl->current_v = bindings.current_v;
  impl->packed_k = bindings.packed_k;
  impl->packed_v = bindings.packed_v;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  output->reset(new AotKvPackCommand(std::move(impl)));
  return Status::Ok();
}

Status AotKvPackCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("AOT KV pack command is not prepared");
  void* arguments[] = {&impl_->prefix_k, &impl_->prefix_v, &impl_->current_k,
                       &impl_->current_v, &impl_->packed_k, &impl_->packed_v};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), 0, cuda_stream, arguments);
}

}  // namespace aginfer::internal
