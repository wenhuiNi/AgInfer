#pragma once

#include "aginfer/c_api.h"
#include "executable_plan.h"
#include "providers/dispatch.h"
#include <memory>
#include <vector>

namespace aginfer::internal {
// Single-profile v2 session. All external addresses are frozen by Prepare.
class ExecutableSession {
 public:
  ExecutableSession(const ParsedExecutablePlan& plan, const std::uint8_t* kernels,
      std::uint64_t kernel_bytes, const std::uint8_t* weights, bool cuda_graph = true);
  ~ExecutableSession();
  Status Bind(std::uint32_t port_id, ai_port_kind kind, const ai_tensor_view& view);
  Status Prepare();
  Status Enqueue(CudaStream stream);
  std::uint32_t PortCount(ai_port_kind kind) const;
  Status PortInfo(ai_port_kind kind, std::uint32_t index, ai_port_info* info) const;
  const ParsedExecutablePlan& plan() const { return plan_; }
  std::uint64_t submitted(std::uint32_t provider_id) const;
  std::uint64_t enqueues() const { return enqueues_; }
  void GraphInfo(ai_cuda_graph_info* info) const;
 private:
  Status PrepareGraph();
  ParsedExecutablePlan plan_;
  const std::uint8_t* kernels_;
  std::uint64_t kernel_bytes_;
  const std::uint8_t* host_weights_;
  std::vector<void*> bindings_;
  std::unique_ptr<CudaDriver> cuda_;
  CuModule module_ = nullptr;
  CuDevicePtr weights_ = 0, arena_ = 0, state_ = 0, workspace_ = 0;
  std::vector<std::unique_ptr<PreparedCommand>> commands_;
  std::vector<std::uint32_t> provider_ids_;
  std::vector<std::uint64_t> submissions_;
  std::uint64_t enqueues_ = 0;
  // Direct submission is retained only for internal diagnostics, not public configuration.
  bool cuda_graph_ = true;
  void* graph_exec_ = nullptr;
  std::uint64_t graph_nodes_ = 0, graph_launches_ = 0;
  bool prepared_ = false;
};
}  // namespace aginfer::internal
