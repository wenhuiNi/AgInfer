#include "providers/aot_prefix_kv_store.h"

#include "prefix_kv_store_payload.h"

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

struct AotPrefixKvStoreCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* key = nullptr;
  const void* value = nullptr;
  void* state_key = nullptr;
  void* state_value = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
};

AotPrefixKvStoreCommand::AotPrefixKvStoreCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

AotPrefixKvStoreCommand::~AotPrefixKvStoreCommand() = default;

Status AotPrefixKvStoreCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotPrefixKvStoreBindings& bindings,
    std::unique_ptr<AotPrefixKvStoreCommand>* output) {
  if (output == nullptr) return Invalid("prefix KV store command output is null");
  output->reset();
  PrefixKvStorePayloadView parsed;
  Status status = ParsePrefixKvStorePayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "prefix KV store target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes ||
      module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "prefix KV store module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("prefix KV store needs a loaded caller-owned CUDA module");
  }
  const std::array<const void*, 4> pointers{
      bindings.key, bindings.value, bindings.state_key, bindings.state_value};
  if (std::any_of(pointers.begin(), pointers.end(), [](const void* pointer) {
        return pointer == nullptr;
      }) ||
      std::any_of(pointers.begin(), pointers.end(), [&](const void* pointer) {
        return !Aligned(pointer, parsed.alignment);
      })) {
    return Invalid("prefix KV store binding is null or misaligned");
  }
  if (bindings.key_bytes < parsed.key_bytes ||
      bindings.value_bytes < parsed.value_bytes ||
      bindings.state_key_bytes < parsed.state_key_bytes ||
      bindings.state_value_bytes < parsed.state_value_bytes) {
    return Invalid("prefix KV store binding is shorter than the exact problem");
  }
  if (Overlaps(bindings.state_key, parsed.state_key_bytes, bindings.state_value,
               parsed.state_value_bytes) ||
      Overlaps(bindings.key, parsed.key_bytes, bindings.state_key,
               parsed.state_key_bytes) ||
      Overlaps(bindings.key, parsed.key_bytes, bindings.state_value,
               parsed.state_value_bytes) ||
      Overlaps(bindings.value, parsed.value_bytes, bindings.state_key,
               parsed.state_key_bytes) ||
      Overlaps(bindings.value, parsed.value_bytes, bindings.state_value,
               parsed.state_value_bytes)) {
    return Invalid("prefix KV store output overlaps an input or another output");
  }
  auto function = driver->GetFunction(module, PrefixKvStoreSymbol());
  if (!function.ok()) return function.status();
  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->key = bindings.key;
  impl->value = bindings.value;
  impl->state_key = bindings.state_key;
  impl->state_value = bindings.state_value;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  output->reset(new AotPrefixKvStoreCommand(std::move(impl)));
  return Status::Ok();
}

Status AotPrefixKvStoreCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("prefix KV store command is not prepared");
  void* arguments[] = {
      &impl_->key, &impl_->value, &impl_->state_key, &impl_->state_value};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), 0, cuda_stream, arguments);
}

}  // namespace aginfer::internal
