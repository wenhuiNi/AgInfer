#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace aginfer::internal {

struct CublasLtLinearBindings {
  const void* x = nullptr;
  const void* weight = nullptr;
  const void* bias = nullptr;
  void* output = nullptr;
  void* workspace = nullptr;
  std::uint64_t workspace_bytes = 0;
};

class CublasLtLinearCommand {
 public:
  ~CublasLtLinearCommand();
  CublasLtLinearCommand(const CublasLtLinearCommand&) = delete;
  CublasLtLinearCommand& operator=(const CublasLtLinearCommand&) = delete;

  static Status Prepare(const std::uint8_t* payload, std::size_t payload_size,
                        const CublasLtLinearBindings& bindings,
                        std::unique_ptr<CublasLtLinearCommand>* output);

  // cuda_stream is a caller-owned cudaStream_t represented as void*. A null
  // value selects CUDA's caller-visible default stream.
  Status Execute(void* cuda_stream);

 private:
  struct Impl;
  explicit CublasLtLinearCommand(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal
