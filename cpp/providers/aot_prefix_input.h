#pragma once

#include "cuda_driver.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct AotPrefixInputBindings {
  std::array<const void*, 3> images{nullptr, nullptr, nullptr};
  std::array<std::uint64_t, 3> image_bytes{0, 0, 0};
  const void* embedding = nullptr;
  std::uint64_t embedding_bytes = 0;
  const void* tokens = nullptr;
  std::uint64_t token_bytes = 0;
  std::array<const void*, 3> image_masks{nullptr, nullptr, nullptr};
  std::array<std::uint64_t, 3> image_mask_bytes{0, 0, 0};
  const void* token_mask = nullptr;
  std::uint64_t token_mask_bytes = 0;
  void* prefix = nullptr;
  std::uint64_t prefix_bytes = 0;
  void* pad_mask = nullptr;
  std::uint64_t pad_mask_bytes = 0;
  void* positions = nullptr;
  std::uint64_t position_bytes = 0;
};

class AotPrefixInputCommand {
 public:
  ~AotPrefixInputCommand();
  AotPrefixInputCommand(const AotPrefixInputCommand&) = delete;
  AotPrefixInputCommand& operator=(const AotPrefixInputCommand&) = delete;

  static Status Prepare(
      const std::uint8_t* payload, std::size_t payload_size,
      std::uint32_t active_arch, std::uint64_t module_bytes,
      const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
      CuModule module, const AotPrefixInputBindings& bindings,
      std::unique_ptr<AotPrefixInputCommand>* output);
  Status Execute(CudaStream cuda_stream);

 private:
  struct Impl;
  explicit AotPrefixInputCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
