#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

// This command consumes the variant's smallest proven BOOL mask boundary and
// writes BSHD directly, fusing mask expansion and the post-attention transpose.
struct FlashInferAttentionBindings {
  const void* query = nullptr;
  std::uint64_t query_bytes = 0;
  const void* key = nullptr;
  std::uint64_t key_bytes = 0;
  const void* value = nullptr;
  std::uint64_t value_bytes = 0;
  const void* mask_2d = nullptr;
  std::uint64_t mask_bytes = 0;
  void* output_bshd = nullptr;
  std::uint64_t output_bytes = 0;
};

class FlashInferAttentionCommand {
 public:
  ~FlashInferAttentionCommand();
  FlashInferAttentionCommand(const FlashInferAttentionCommand&) = delete;
  FlashInferAttentionCommand& operator=(const FlashInferAttentionCommand&) = delete;

  static Status Prepare(const std::uint8_t* payload, std::size_t payload_size,
                        std::uint32_t active_arch,
                        const FlashInferAttentionBindings& bindings,
                        std::unique_ptr<FlashInferAttentionCommand>* output);

  // Execute performs one fixed native kernel launch on the caller-owned stream.
  Status Execute(void* cuda_stream);

 private:
  struct Impl;
  explicit FlashInferAttentionCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
