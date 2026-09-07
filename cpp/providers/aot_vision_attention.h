#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotVisionAttentionBindings {
  const void* query = nullptr;
  std::uint64_t query_bytes = 0;
  const void* key = nullptr;
  std::uint64_t key_bytes = 0;
  const void* value = nullptr;
  std::uint64_t value_bytes = 0;
  const void* mask = nullptr;
  std::uint64_t mask_bytes = 0;
  void* output = nullptr;
  std::uint64_t output_bytes = 0;
};

class AotVisionAttentionCommand {
 public:
  ~AotVisionAttentionCommand();
  AotVisionAttentionCommand(const AotVisionAttentionCommand&) = delete;
  AotVisionAttentionCommand& operator=(const AotVisionAttentionCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotVisionAttentionBindings& bindings,
      std::unique_ptr<AotVisionAttentionCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotVisionAttentionCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
