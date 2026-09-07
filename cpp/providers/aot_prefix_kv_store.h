#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotPrefixKvStoreBindings {
  const void* key = nullptr;
  std::uint64_t key_bytes = 0;
  const void* value = nullptr;
  std::uint64_t value_bytes = 0;
  void* state_key = nullptr;
  std::uint64_t state_key_bytes = 0;
  void* state_value = nullptr;
  std::uint64_t state_value_bytes = 0;
};

class AotPrefixKvStoreCommand {
 public:
  ~AotPrefixKvStoreCommand();
  AotPrefixKvStoreCommand(const AotPrefixKvStoreCommand&) = delete;
  AotPrefixKvStoreCommand& operator=(const AotPrefixKvStoreCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotPrefixKvStoreBindings& bindings,
      std::unique_ptr<AotPrefixKvStoreCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotPrefixKvStoreCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
