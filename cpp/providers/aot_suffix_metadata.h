#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotSuffixMetadataBindings {
  const void* prefix_pad = nullptr;
  std::uint64_t prefix_pad_bytes = 0;
  void* attention_mask = nullptr;
  std::uint64_t attention_mask_bytes = 0;
  void* positions = nullptr;
  std::uint64_t positions_bytes = 0;
};

class AotSuffixMetadataCommand {
 public:
  ~AotSuffixMetadataCommand();
  AotSuffixMetadataCommand(const AotSuffixMetadataCommand&) = delete;
  AotSuffixMetadataCommand& operator=(const AotSuffixMetadataCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotSuffixMetadataBindings& bindings,
      std::unique_ptr<AotSuffixMetadataCommand>* output);
  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotSuffixMetadataCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
