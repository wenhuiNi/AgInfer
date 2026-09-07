#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotActivationBindings {
  const void* input = nullptr;
  std::uint64_t input_bytes = 0;
  void* output = nullptr;
  std::uint64_t output_bytes = 0;
};

class AotActivationCommand {
 public:
  ~AotActivationCommand();
  AotActivationCommand(const AotActivationCommand&) = delete;
  AotActivationCommand& operator=(const AotActivationCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotActivationBindings& bindings,
      std::unique_ptr<AotActivationCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotActivationCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
