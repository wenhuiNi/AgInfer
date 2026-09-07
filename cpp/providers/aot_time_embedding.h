#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotTimeEmbeddingBindings {
  const void* timestep = nullptr;
  std::uint64_t timestep_bytes = 0;
  void* output = nullptr;
  std::uint64_t output_bytes = 0;
};

class AotTimeEmbeddingCommand {
 public:
  ~AotTimeEmbeddingCommand();
  AotTimeEmbeddingCommand(const AotTimeEmbeddingCommand&) = delete;
  AotTimeEmbeddingCommand& operator=(const AotTimeEmbeddingCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotTimeEmbeddingBindings& bindings,
      std::unique_ptr<AotTimeEmbeddingCommand>* output);
  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotTimeEmbeddingCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
