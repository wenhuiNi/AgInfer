#include "providers/patch_projection.h"

#include "patch_projection_payload.h"
#include "providers/cublaslt_linear.h"

#include <array>
#include <cstdint>
#include <limits>
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

struct PatchProjectionCommand::Impl {
  CudaDriver* driver = nullptr;
  CuFunction patchify = nullptr;
  PatchProjectionBindings bindings;
  std::array<std::uint32_t, 3> grid{1, 1, 1};
  std::array<std::uint32_t, 3> block{1, 1, 1};
  std::unique_ptr<CublasLtLinearCommand> linear;
};

PatchProjectionCommand::PatchProjectionCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PatchProjectionCommand::~PatchProjectionCommand() = default;

Status PatchProjectionCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, std::uint64_t module_bytes,
    const std::array<std::uint8_t, 32>& module_sha256, CudaDriver* driver,
    CuModule module, const PatchProjectionBindings& bindings,
    std::unique_ptr<PatchProjectionCommand>* output) {
  if (output == nullptr) return Invalid("patch projection command output is null");
  output->reset();
  PatchProjectionPayloadView parsed;
  Status status = ParsePatchProjectionPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "patch projection target does not match the active device");
  }
  if (module_bytes != parsed.module_bytes || module_sha256 != parsed.module_sha256) {
    return Status(StatusCode::kIncompatibleAbi,
                  "patch projection module identity does not match its payload");
  }
  if (driver == nullptr || module == nullptr) {
    return Invalid("patch projection needs a loaded caller-owned CUDA module");
  }
  if (bindings.image == nullptr || bindings.weight == nullptr ||
      bindings.bias == nullptr || bindings.output == nullptr ||
      bindings.workspace == nullptr ||
      !Aligned(bindings.image, parsed.image_alignment) ||
      !Aligned(bindings.weight, parsed.weight_alignment) ||
      !Aligned(bindings.bias, parsed.bias_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment) ||
      !Aligned(bindings.workspace, parsed.workspace_alignment)) {
    return Invalid("patch projection binding is null or misaligned");
  }
  if (parsed.patch_workspace_bytes >
      std::numeric_limits<std::uint64_t>::max() - parsed.cublaslt.workspace_bytes) {
    return Invalid("patch projection workspace size overflows");
  }
  const std::uint64_t workspace_bytes =
      parsed.patch_workspace_bytes + parsed.cublaslt.workspace_bytes;
  if (bindings.image_bytes < parsed.image_bytes ||
      bindings.weight_bytes < parsed.weight_bytes ||
      bindings.bias_bytes < parsed.bias_bytes ||
      bindings.output_bytes < parsed.output_bytes ||
      bindings.workspace_bytes != workspace_bytes) {
    return Invalid("patch projection binding is shorter or workspace is not exact");
  }
  const std::array<const void*, 3> reads{
      bindings.image, bindings.weight, bindings.bias};
  const std::array<std::uint64_t, 3> read_bytes{
      parsed.image_bytes, parsed.weight_bytes, parsed.bias_bytes};
  for (std::size_t index = 0; index < reads.size(); ++index) {
    if (Overlaps(bindings.output, parsed.output_bytes, reads[index],
                 read_bytes[index]) ||
        Overlaps(bindings.workspace, workspace_bytes, reads[index],
                 read_bytes[index])) {
      return Invalid("patch projection read and write/workspace buffers overlap");
    }
  }
  if (Overlaps(bindings.output, parsed.output_bytes, bindings.workspace,
               workspace_bytes)) {
    return Invalid("patch projection output and workspace overlap");
  }
  auto function = driver->GetFunction(module, PatchProjectionSymbol());
  if (!function.ok()) return function.status();

  auto* workspace = static_cast<std::uint8_t*>(bindings.workspace);
  CublasLtLinearBindings linear_bindings;
  linear_bindings.x = workspace;
  linear_bindings.weight = bindings.weight;
  linear_bindings.bias = bindings.bias;
  linear_bindings.output = bindings.output;
  if (parsed.cublaslt.workspace_bytes != 0) {
    linear_bindings.workspace = workspace + parsed.patch_workspace_bytes;
    linear_bindings.workspace_bytes = parsed.cublaslt.workspace_bytes;
  }
  std::unique_ptr<CublasLtLinearCommand> linear;
  status = CublasLtLinearCommand::Prepare(payload + 192, 192,
                                          linear_bindings, &linear);
  if (!status.ok()) return status;

  auto impl = std::make_unique<Impl>();
  impl->driver = driver;
  impl->patchify = std::move(function).value();
  impl->bindings = bindings;
  impl->grid = parsed.grid;
  impl->block = parsed.block;
  impl->linear = std::move(linear);
  output->reset(new PatchProjectionCommand(std::move(impl)));
  return Status::Ok();
}

Status PatchProjectionCommand::Execute(CudaStream cuda_stream) {
  if (impl_ == nullptr) return Invalid("patch projection command is not prepared");
  void* image = const_cast<void*>(impl_->bindings.image);
  void* patches = impl_->bindings.workspace;
  void* arguments[] = {&image, &patches};
  Status status = impl_->driver->Launch(
      impl_->patchify, impl_->grid.data(), impl_->block.data(), 0,
      cuda_stream, arguments);
  if (!status.ok()) return status;
  return impl_->linear->Execute(cuda_stream);
}

}  // namespace aginfer::internal
