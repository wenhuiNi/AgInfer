#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotKvPackBindings {
  const void* prefix_k = nullptr;
  std::uint64_t prefix_k_bytes = 0;
  const void* prefix_v = nullptr;
  std::uint64_t prefix_v_bytes = 0;
  const void* current_k = nullptr;
  std::uint64_t current_k_bytes = 0;
  const void* current_v = nullptr;
  std::uint64_t current_v_bytes = 0;
  void* packed_k = nullptr;
  std::uint64_t packed_k_bytes = 0;
  void* packed_v = nullptr;
  std::uint64_t packed_v_bytes = 0;
};

class AotKvPackCommand {
 public:
  ~AotKvPackCommand();
  AotKvPackCommand(const AotKvPackCommand&) = delete;
  AotKvPackCommand& operator=(const AotKvPackCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotKvPackBindings& bindings,
      std::unique_ptr<AotKvPackCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotKvPackCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
