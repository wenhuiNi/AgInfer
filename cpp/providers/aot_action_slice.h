#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotActionSliceBindings {
  const void* input = nullptr;
  std::uint64_t input_bytes = 0;
  void* output = nullptr;
  std::uint64_t output_bytes = 0;
};

class AotActionSliceCommand {
 public:
  ~AotActionSliceCommand();
  AotActionSliceCommand(const AotActionSliceCommand&) = delete;
  AotActionSliceCommand& operator=(const AotActionSliceCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotActionSliceBindings& bindings,
      std::unique_ptr<AotActionSliceCommand>* output);
  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotActionSliceCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
