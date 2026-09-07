#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotPointwiseBindings {
  const void* lhs = nullptr;
  std::uint64_t lhs_bytes = 0;
  const void* rhs = nullptr;
  std::uint64_t rhs_bytes = 0;
  void* output = nullptr;
  std::uint64_t output_bytes = 0;
};

class AotPointwiseCommand {
 public:
  ~AotPointwiseCommand();
  AotPointwiseCommand(const AotPointwiseCommand&) = delete;
  AotPointwiseCommand& operator=(const AotPointwiseCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotPointwiseBindings& bindings,
      std::unique_ptr<AotPointwiseCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotPointwiseCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
