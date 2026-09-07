#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct PatchProjectionBindings {
  const void* image = nullptr;
  std::uint64_t image_bytes = 0;
  const void* weight = nullptr;
  std::uint64_t weight_bytes = 0;
  const void* bias = nullptr;
  std::uint64_t bias_bytes = 0;
  void* output = nullptr;
  std::uint64_t output_bytes = 0;
  void* workspace = nullptr;
  std::uint64_t workspace_bytes = 0;
};

class PatchProjectionCommand {
 public:
  ~PatchProjectionCommand();
  PatchProjectionCommand(const PatchProjectionCommand&) = delete;
  PatchProjectionCommand& operator=(const PatchProjectionCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const PatchProjectionBindings& bindings,
      std::unique_ptr<PatchProjectionCommand>* output);
  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit PatchProjectionCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
