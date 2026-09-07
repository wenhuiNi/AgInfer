#include "providers/aot_time_embedding.h"

#include "time_embedding_payload.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace aginfer::internal {
namespace {

Status Invalid(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}
bool Aligned(const void* p, std::uint32_t a) {
  return reinterpret_cast<std::uintptr_t>(p) % a == 0;
}
bool Overlaps(const void* a, std::uint64_t an, const void* b,
              std::uint64_t bn) {
  const auto av = reinterpret_cast<std::uintptr_t>(a);
  const auto bv = reinterpret_cast<std::uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}

}  // namespace

struct AotTimeEmbeddingCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  const void* timestep = nullptr;
  void* output = nullptr;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
};

AotTimeEmbeddingCommand::AotTimeEmbeddingCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AotTimeEmbeddingCommand::~AotTimeEmbeddingCommand() = default;

Status AotTimeEmbeddingCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const AotTimeEmbeddingBindings& bindings,
    std::unique_ptr<AotTimeEmbeddingCommand>* output) {
  if (output == nullptr) return Invalid("time embedding command output is null");
  output->reset();
  TimeEmbeddingPayloadView parsed;
  Status status = ParseTimeEmbeddingPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "time embedding target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes || module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "time embedding module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("time embedding needs a loaded caller-owned CUDA module");
  }
  if (bindings.timestep == nullptr || bindings.output == nullptr ||
      !Aligned(bindings.timestep, parsed.input_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment)) {
    return Invalid("time embedding binding is null or misaligned");
  }
  if (bindings.timestep_bytes < parsed.input_bytes ||
      bindings.output_bytes < parsed.output_bytes) {
    return Invalid("time embedding binding is shorter than the exact problem");
  }
  if (Overlaps(bindings.timestep, parsed.input_bytes, bindings.output,
               parsed.output_bytes)) {
    return Invalid("time embedding input and output overlap");
  }
  auto function = driver->GetFunction(module, TimeEmbeddingSymbol());
  if (!function.ok()) return function.status();
  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->function = std::move(function).value();
  impl->timestep = bindings.timestep;
  impl->output = bindings.output;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  output->reset(new AotTimeEmbeddingCommand(std::move(impl)));
  return Status::Ok();
}

Status AotTimeEmbeddingCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("time embedding command is not prepared");
  void* arguments[] = {&impl_->timestep, &impl_->output};
  return impl_->driver->Launch(impl_->function, impl_->grid.data(),
                               impl_->block.data(), 0, cuda_stream, arguments);
}

}  // namespace aginfer::internal
