#include "providers/projection_split.h"
#include "projection_split_payload.h"

namespace aginfer::internal {
namespace {
class ProjectionSplit final : public PreparedCommand {
 public:
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  std::array<void*, 7> pointers{};
  ProjectionSplitPayloadView p;
  std::array<std::uint32_t, 3> grid{}, block{256, 1, 1};
  Status Execute(CudaStream stream) override {
    if (p.rope_pack) {
      void* args[] = {&pointers[0], &pointers[1], &pointers[2], &pointers[3],
                      &pointers[4], &pointers[5], &pointers[6]};
      return driver->Launch(function, grid.data(), block.data(), 0, stream, args);
    }
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
      || !module.driver || !module.module || b.size() != (p.rope_pack ? 7 : 4)) return bad();
  std::array<std::uint64_t, 7> bytes{};
  if (p.rope_pack) bytes = {256000, 200, 495616, 495616, 204800, 521216, 521216};
  else for (int i = 0; i < 3; ++i) { bytes[i + 1] = std::uint64_t(p.rows) * p.widths[i] * 2; bytes[0] += bytes[i + 1]; }
  const int count = p.rope_pack ? 7 : 4;
  const int reads = p.rope_pack ? 4 : 1;
  for (int i = 0; i < count; ++i) {
    if (!b[i].data || reinterpret_cast<std::uintptr_t>(b[i].data) % 16 || b[i].bytes < bytes[i]
        || b[i].access != (i < reads ? CommandOperandAccess::kRead : CommandOperandAccess::kWrite)) return bad();
    for (int j = 0; j < i; ++j) {
      // Read-only aliases (e.g. identical prefix K/V) are harmless. Only
      // overlaps involving an output can violate this out-of-place form.
      if (i < reads) continue;
      auto x = reinterpret_cast<std::uintptr_t>(b[i].data), y = reinterpret_cast<std::uintptr_t>(b[j].data);
      if (x <= y ? y - x < bytes[i] : x - y < bytes[j]) return bad();
    }
  }
  auto function = module.driver->GetFunction(module.module, p.rope_pack
      ? "aginfer_qkv_rope_pack_bf16_s50_p968_h8_d256" : "aginfer_projection_split_bf16");
  if (!function.ok()) return function.status();
  auto command = std::make_unique<ProjectionSplit>();
  command->driver = module.driver; command->function = function.value(); command->p = p;
  for (int i = 0; i < count; ++i) command->pointers[i] = b[i].data;
  // 225 rotary blocks, 242 prefix-copy blocks, 7 suffix-value blocks.
  command->grid = {p.rope_pack ? 474U : static_cast<std::uint32_t>((bytes[0] / 16 + 255) / 256), 1, 1};
  *out = std::move(command);
  return Status::Ok();
}
}  // namespace aginfer::internal
