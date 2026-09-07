#include "providers/projection_split.h"
#include "projection_split_payload.h"

namespace aginfer::internal {
namespace {
class ProjectionSplit final : public PreparedCommand {
 public:
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  std::array<void*, 4> pointers{};
  ProjectionSplitPayloadView p;
  std::array<std::uint32_t, 3> grid{}, block{256, 1, 1};
  Status Execute(CudaStream stream) override {
    void* args[] = {&pointers[0], &pointers[1], &pointers[2], &pointers[3],
                    &p.rows, &p.widths[0], &p.widths[1], &p.widths[2]};
    return driver->Launch(function, grid.data(), block.data(), 0, stream, args);
  }
};
}
Status PrepareProjectionSplit(std::span<const std::uint8_t> payload, std::span<const CommandBuffer> b,
                              const CommandModule& module, std::unique_ptr<PreparedCommand>* out) {
  auto bad = [] { return Status(StatusCode::kInvalidArgument, "projection split binding contract mismatch"); };
  if (!out) return bad();
  out->reset();
  ProjectionSplitPayloadView p;
  auto status = ParseProjectionSplitPayload(payload.data(), payload.size(), &p);
  if (!status.ok()) return status;
  if (module.arch != 120 || module.bytes != p.module_bytes || module.sha256 != p.module_sha256
      || !module.driver || !module.module || b.size() != 4) return bad();
  std::array<std::uint64_t, 4> bytes{};
  for (int i = 0; i < 3; ++i) { bytes[i + 1] = std::uint64_t(p.rows) * p.widths[i] * 2; bytes[0] += bytes[i + 1]; }
  for (int i = 0; i < 4; ++i) {
    if (!b[i].data || reinterpret_cast<std::uintptr_t>(b[i].data) % 16 || b[i].bytes < bytes[i]
        || b[i].access != (i == 0 ? CommandOperandAccess::kRead : CommandOperandAccess::kWrite)) return bad();
    for (int j = 0; j < i; ++j) {
      auto x = reinterpret_cast<std::uintptr_t>(b[i].data), y = reinterpret_cast<std::uintptr_t>(b[j].data);
      if (x <= y ? y - x < bytes[i] : x - y < bytes[j]) return bad();
    }
  }
  auto function = module.driver->GetFunction(module.module, "aginfer_projection_split_bf16");
  if (!function.ok()) return function.status();
  auto command = std::make_unique<ProjectionSplit>();
  command->driver = module.driver; command->function = function.value(); command->p = p;
  for (int i = 0; i < 4; ++i) command->pointers[i] = b[i].data;
  command->grid = {static_cast<std::uint32_t>((bytes[0] / 16 + 255) / 256), 1, 1};
  *out = std::move(command);
  return Status::Ok();
}
}  // namespace aginfer::internal
