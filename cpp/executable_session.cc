#include "executable_session.h"
#include "sha256.h"
#include <algorithm>
#include <limits>
#include <string>
#ifdef AGINFER_HAS_CUDA_PROVIDERS
#include <cuda_runtime_api.h>
#endif

namespace aginfer::internal {
namespace {
Status Invalid(const std::string& text) { return Status(StatusCode::kInvalidArgument, text); }
ai_dtype PublicDType(ExecutableDType type) {
  switch (type) {
    case ExecutableDType::kFp32: return AI_DTYPE_F32;
    case ExecutableDType::kFp16: return AI_DTYPE_F16;
    case ExecutableDType::kBf16: return AI_DTYPE_BF16;
    case ExecutableDType::kInt32: return AI_DTYPE_I32;
    case ExecutableDType::kBool: return AI_DTYPE_BOOL;
  }
  return 0;
}
bool Overlap(const CommandBuffer& a, const CommandBuffer& b) {
  auto x = reinterpret_cast<std::uintptr_t>(a.data), y = reinterpret_cast<std::uintptr_t>(b.data);
  return x <= y ? y - x < a.bytes : x - y < b.bytes;
}
#ifdef AGINFER_HAS_CUDA_PROVIDERS
Status PreflightLibraries(const std::vector<std::unique_ptr<PreparedCommand>>& commands) {
  if (std::none_of(commands.begin(), commands.end(),
      [](const auto& c){return c->NeedsLibraryPreflight();})) return Status::Ok();
  struct Capture {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    ~Capture(){if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
  } capture;
  auto checked=[](cudaError_t error) {
    return error==cudaSuccess ? Status::Ok() : Status(StatusCode::kCudaError,
        std::string("library initialization preflight: ")+cudaGetErrorString(error));
  };
  auto status=checked(cudaStreamCreateWithFlags(&capture.stream,cudaStreamNonBlocking));
  if(!status.ok())return status;
  status=checked(cudaStreamBeginCapture(capture.stream,cudaStreamCaptureModeThreadLocal));
  if(!status.ok())return status;
  // Capture constructs nodes only: input contents are never read and no model
  // state/output is modified. It must not increment the submission ledger.
  for(const auto& command:commands) {
    if(!command->NeedsLibraryPreflight())continue;
    status=command->Execute(capture.stream);
    if(!status.ok())break;
  }
  const auto ended=checked(cudaStreamEndCapture(capture.stream,&capture.graph));
  if(!status.ok())return status;
  return ended;
}
#endif
}  // namespace

ExecutableSession::ExecutableSession(const ParsedExecutablePlan& plan,
    const std::uint8_t* kernels, std::uint64_t kernel_bytes, const std::uint8_t* weights)
    : plan_(plan), kernels_(kernels), kernel_bytes_(kernel_bytes), host_weights_(weights),
      bindings_(plan.port_count, nullptr) {}

ExecutableSession::~ExecutableSession() {
  if (!cuda_) return;
  cuda_->MakeCurrent();
  commands_.clear();
  cuda_->Free(workspace_); cuda_->Free(state_); cuda_->Free(arena_); cuda_->Free(weights_);
  cuda_->UnloadModule(module_);
}

std::uint32_t ExecutableSession::PortCount(ai_port_kind kind) const {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < plan_.port_count; ++i) {
    ExecutablePortView p; plan_.ReadPort(i, &p);
    if (static_cast<std::uint32_t>(p.kind) == kind) ++count;
  }
  return count;
}

Status ExecutableSession::PortInfo(ai_port_kind kind, std::uint32_t index, ai_port_info* info) const {
  std::uint32_t matched = 0;
  for (std::uint32_t i = 0; i < plan_.port_count; ++i) {
    ExecutablePortView p; plan_.ReadPort(i, &p);
    if (static_cast<std::uint32_t>(p.kind) != kind || matched++ != index) continue;
    ExecutableValueView v; plan_.ReadValue(p.value_id, &v);
    const auto caller_size = info->struct_size;
    ai_port_info_init(info); info->struct_size = caller_size;
    info->port_id = p.port_id; info->kind = kind; info->dtype = PublicDType(v.dtype);
    info->location = AI_MEMORY_DEVICE; info->rank = v.rank; info->byte_size = v.byte_size;
    info->diagnostic_name = "";
    std::int64_t stride = 1;
    for (std::uint32_t d = v.rank; d-- > 0;) {
      info->shape[d] = v.shape[d]; info->stride[d] = stride;
      if (d != 0) {
        if (v.shape[d] > std::numeric_limits<std::int64_t>::max() / stride)
          return Invalid("executable port stride overflows");
        stride *= v.shape[d];
      }
    }
    return Status::Ok();
  }
  return Invalid("port index is outside the executable profile");
}

Status ExecutableSession::Bind(std::uint32_t id, ai_port_kind kind, const ai_tensor_view& view) {
  if (view.flags != 0 || view.data == nullptr || view.shape == nullptr || view.stride == nullptr)
    return Invalid("invalid executable port view");
  for (std::uint32_t i = 0; i < PortCount(kind); ++i) {
    ai_port_info expected; ai_port_info_init(&expected);
    auto status = PortInfo(kind, i, &expected);
    if (!status.ok()) return status;
    if (expected.port_id != id) continue;
    if (view.dtype != expected.dtype || view.location != expected.location ||
        view.rank != expected.rank || view.byte_size < expected.byte_size)
      return Invalid("executable port dtype/location/rank/size mismatch");
    for (std::uint32_t d = 0; d < expected.rank; ++d)
      if (view.shape[d] != expected.shape[d] || view.stride[d] != expected.stride[d])
        return Invalid("executable port shape or stride mismatch");
    if (cuda_ && bindings_[id] != view.data)
      return Status(StatusCode::kInvalidState, "executable port addresses are frozen once Prepare begins");
    bindings_[id] = view.data;
    return Status::Ok();
  }
  return Invalid("unknown executable port ID or kind");
}

Status ExecutableSession::Prepare() {
  if (prepared_) return cuda_->MakeCurrent();
  if (cuda_) return Status(StatusCode::kInvalidState, "failed executable Prepare requires a new session");
  if (std::any_of(bindings_.begin(), bindings_.end(), [](void* p) { return p == nullptr; }))
    return Status(StatusCode::kInvalidState, "bind all executable ports before Prepare");
  auto created = CudaDriver::Create(plan_.target_arch);
  if (!created.ok()) return created.status();
  cuda_ = std::make_unique<CudaDriver>(std::move(created).value());
  auto loaded = cuda_->LoadModule(kernels_);
  if (!loaded.ok()) return loaded.status();
  module_ = loaded.value();
  for (auto [size, pointer] : {std::pair{plan_.weights_bytes, &weights_},
       {plan_.arena_bytes, &arena_}, {plan_.state_bytes, &state_}, {plan_.workspace_bytes, &workspace_}}) {
    auto allocation = cuda_->Allocate(size);
    if (!allocation.ok()) return allocation.status();
    *pointer = allocation.value();
  }
  auto status = cuda_->CopyHostToDevice(weights_, host_weights_, plan_.weights_bytes);
  if (!status.ok()) return status;
  status = cuda_->Zero(state_, plan_.state_bytes);
  if (!status.ok()) return status;
  std::vector<CommandBuffer> values(plan_.value_count);
  for (std::uint32_t i = 0; i < plan_.value_count; ++i) {
    ExecutableValueView v; plan_.ReadValue(i, &v);
    CuDevicePtr base = 0;
    switch (v.region) {
      case ExecutableValueRegion::kWeights: base = weights_; break;
      case ExecutableValueRegion::kArena: base = arena_; break;
      case ExecutableValueRegion::kState: base = state_; break;
      default: break;
    }
    if (base) values[i] = {reinterpret_cast<void*>(base + v.offset), v.byte_size};
  }
  for (std::uint32_t i = 0; i < plan_.port_count; ++i) {
    ExecutablePortView p; plan_.ReadPort(i, &p);
    std::uint32_t root; plan_.RootValue(p.value_id, &root);
    ExecutableValueView v; plan_.ReadValue(root, &v);
    values[root] = {bindings_[p.port_id], v.byte_size};
  }
  for (std::uint32_t i = 0; i < plan_.value_count; ++i) {
    std::uint32_t root; plan_.RootValue(i, &root);
    if (root != i) values[i] = values[root];
  }
  Sha256 hash; hash.Update(kernels_, kernel_bytes_);
  CommandModule module{plan_.target_arch, kernel_bytes_, hash.Final(), cuda_.get(), module_};
  const auto& stream = plan_.command_stream;
  commands_.reserve(stream.command_count); provider_ids_.reserve(stream.command_count);
  submissions_.assign(stream.command_count, 0);
  for (std::uint32_t i = 0; i < stream.command_count; ++i) {
    CommandRecordView record; stream.ReadCommand(i, &record);
    std::vector<CommandBuffer> operands;
    operands.reserve(record.operand_count);
    for (std::uint32_t j = 0; j < record.operand_count; ++j) {
      CommandOperandView op; stream.ReadOperand(record.first_operand + j, &op);
      auto b = values[op.value_id];
      if (b.data == nullptr || op.byte_offset >= b.bytes)
        return Invalid("command references unavailable value storage");
      b.data = static_cast<std::uint8_t*>(b.data) + op.byte_offset;
      b.bytes -= op.byte_offset; b.access = op.access;
      if (op.access != CommandOperandAccess::kRead) {
        std::uint32_t root; plan_.RootValue(op.value_id, &root);
        ExecutableValueView v; plan_.ReadValue(root, &v);
        if (v.region == ExecutableValueRegion::kWeights || v.region == ExecutableValueRegion::kExternalInput)
          return Invalid("command attempts to write immutable storage");
      }
      operands.push_back(b);
    }
    for (std::size_t a = 0; a < operands.size(); ++a)
      for (std::size_t b = a + 1; b < operands.size(); ++b)
        if ((operands[a].access != CommandOperandAccess::kRead || operands[b].access != CommandOperandAccess::kRead)
            && Overlap(operands[a], operands[b]))
          return Invalid("command input/output ranges overlap");
    std::span<const std::uint8_t> payload; stream.Payload(i, &payload);
    CommandBuffer workspace{record.workspace_bytes ? reinterpret_cast<void*>(workspace_ + record.workspace_offset) : nullptr,
                            record.workspace_bytes};
    std::unique_ptr<PreparedCommand> command;
    status = PrepareProviderCommand(record, payload, operands, workspace, module, &command);
    if (!status.ok()) return Status(status.code(), "command " + std::to_string(i) + ": " + status.message());
    commands_.push_back(std::move(command)); provider_ids_.push_back(record.provider_id);
  }
#ifdef AGINFER_HAS_CUDA_PROVIDERS
  status=PreflightLibraries(commands_);
  if(!status.ok())return status;
#endif
  prepared_ = true;
  return Status::Ok();
}

Status ExecutableSession::Enqueue(CudaStream stream) {
  if (!prepared_) return Status(StatusCode::kInvalidState, "executable session must be prepared");
  // No AgInfer parsing/lookup, allocation, IO, tactic selection or synchronization.
  // Prepared vendor calls may still query their cached kernel handles.
  for (std::size_t i = 0; i < commands_.size(); ++i) {
    auto status = commands_[i]->Execute(stream);
    if (!status.ok()) return status;
    ++submissions_[i];
  }
  ++enqueues_;
  return Status::Ok();
}
std::uint64_t ExecutableSession::submitted(std::uint32_t provider) const {
  std::uint64_t count = 0;
  for (std::size_t i = 0; i < provider_ids_.size(); ++i)
    if (provider == 0 || provider_ids_[i] == provider) count += submissions_[i];
  return count;
}
}  // namespace aginfer::internal
