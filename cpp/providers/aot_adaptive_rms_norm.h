#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotAdaptiveRmsNormBindings {
  const void* hidden = nullptr;
  std::uint64_t hidden_bytes = 0;
  const void* modulation = nullptr;
  std::uint64_t modulation_bytes = 0;
  void* normalized = nullptr;
  std::uint64_t normalized_bytes = 0;
  void* gate = nullptr;
  std::uint64_t gate_bytes = 0;
};

class AotAdaptiveRmsNormCommand {
 public:
  ~AotAdaptiveRmsNormCommand();
  AotAdaptiveRmsNormCommand(const AotAdaptiveRmsNormCommand&) = delete;
  AotAdaptiveRmsNormCommand& operator=(const AotAdaptiveRmsNormCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotAdaptiveRmsNormBindings& bindings,
      std::unique_ptr<AotAdaptiveRmsNormCommand>* output);

  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotAdaptiveRmsNormCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
