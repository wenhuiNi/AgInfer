#pragma once

#include "command_stream.h"
#include "cuda_driver.h"
#include "status.h"
#include <array>
#include <memory>
#include <span>

namespace aginfer::internal {
struct CommandBuffer {
  void* data = nullptr;
  std::uint64_t bytes = 0;
  CommandOperandAccess access = CommandOperandAccess::kRead;
};
struct CommandModule {
  std::uint32_t arch = 0;
  std::uint64_t bytes = 0;
  std::array<std::uint8_t, 32> sha256{};
  CudaDriver* driver = nullptr;
  CuModule module = nullptr;
};
class PreparedCommand {
 public:
  virtual ~PreparedCommand() = default;
  // Library calls may defer module initialization until their first submission.
  // Prepare records (but never launches) these calls once to finish that work.
  virtual bool NeedsLibraryPreflight() const { return false; }
  virtual Status Execute(CudaStream stream) = 0;
};
Status PrepareProviderCommand(const CommandRecordView& record,
    std::span<const std::uint8_t> payload, std::span<const CommandBuffer> buffers,
    const CommandBuffer& workspace, const CommandModule& module,
    std::unique_ptr<PreparedCommand>* output);
}  // namespace aginfer::internal
