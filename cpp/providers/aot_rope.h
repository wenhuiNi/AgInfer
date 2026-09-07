#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotRopeBindings {
  const void* input_bshd = nullptr;
  std::uint64_t input_bytes = 0;
  const void* positions = nullptr;
  std::uint64_t position_bytes = 0;
  void* output_bhsd = nullptr;
  std::uint64_t output_bytes = 0;
};

class AotRopeCommand {
 public:
  ~AotRopeCommand();
  AotRopeCommand(const AotRopeCommand&) = delete;
  AotRopeCommand& operator=(const AotRopeCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotRopeBindings& bindings,
      std::unique_ptr<AotRopeCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotRopeCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
