#include "providers/gelu_mul.h"
#include "gelu_mul_payload.h"
#include <algorithm>

namespace aginfer::internal {
namespace {
class GeluMul final : public PreparedCommand {
 public:
  CudaDriver* driver = nullptr;
  CuFunction function = nullptr;
  std::array<void*, 3> pointers{};
  std::uint64_t numel = 0;
  std::uint32_t packed_width = 0;
  std::array<std::uint32_t, 3> grid{}, block{256, 1, 1};
  Status Execute(CudaStream stream) override {
    if (packed_width) {
      void* args[] = {&pointers[0], &pointers[1], &numel, &packed_width};
      return driver->Launch(function, grid.data(), block.data(), 0, stream, args);
    }
    void* args[] = {&pointers[0], &pointers[1], &pointers[2], &numel};
    return driver->Launch(function, grid.data(), block.data(), 0, stream, args);
  }
};
}
Status PrepareGeluMul(std::span<const std::uint8_t> payload, std::span<const CommandBuffer> b,
                      const CommandModule& module, std::unique_ptr<PreparedCommand>* out) {
  auto bad = [] { return Status(StatusCode::kInvalidArgument, "GELU-multiply binding contract mismatch"); };
  if (!out) return bad();
  out->reset();
  GeluMulPayloadView p;
  auto status = ParseGeluMulPayload(payload.data(), payload.size(), &p);
  if (!status.ok()) return status;
  const int count = p.packed_width ? 2 : 3;
  if (module.arch != 120 || module.bytes != p.module_bytes || module.sha256 != p.module_sha256
      || !module.driver || !module.module || b.size() != static_cast<std::size_t>(count)) return bad();
  const auto bytes = p.numel * 2;
  for (int i = 0; i < count; ++i) {
    auto size = bytes * ((p.packed_width && i == 0) ? 2 : 1);
    if (!b[i].data || reinterpret_cast<std::uintptr_t>(b[i].data) % 2 || b[i].bytes < size
        || b[i].access != (i == count-1 ? CommandOperandAccess::kWrite : CommandOperandAccess::kRead)) return bad();
  }
  // Read-only inputs may alias; the output may overlap neither input.
  auto output = reinterpret_cast<std::uintptr_t>(b[count-1].data);
  for (int i = 0; i < count-1; ++i) {
    auto input = reinterpret_cast<std::uintptr_t>(b[i].data);
    if (output <= input ? input - output < bytes : output - input < bytes*(p.packed_width ? 2 : 1)) return bad();
  }
  auto function = module.driver->GetFunction(module.module, p.packed_tiled
      ? (p.numel / p.packed_width > 128
         ? "aginfer_gelu_tanh_mul_packed_prefill_bf16"
         : "aginfer_gelu_tanh_mul_packed_tiled_bf16")
      : p.packed_width ? "aginfer_gelu_tanh_mul_packed_bf16" : "aginfer_gelu_tanh_mul_bf16");
  if (!function.ok()) return function.status();
  auto command = std::make_unique<GeluMul>();
  command->driver = module.driver; command->function = function.value(); command->numel = p.numel;
  command->packed_width = p.packed_width;
  for (int i = 0; i < count; ++i) command->pointers[i] = b[i].data;
  command->grid = {static_cast<std::uint32_t>(std::min<std::uint64_t>((p.numel + 255) / 256, 4096)), 1, 1};
  if (p.packed_tiled) {
    const auto columns = (p.packed_width + 255) / 256;
    // Large prefill retains column tiling but iterates rows in each CTA.
    // Preserve ALL old <=128-row launches: older AIM modules do not stride rows.
    const auto rows = p.numel / p.packed_width;
    command->grid = {columns, static_cast<std::uint32_t>(rows <= 128 ? rows :
        std::min<std::uint64_t>(rows, 4096 / columns)), 1};
  }
  *out = std::move(command);
  return Status::Ok();
}
}  // namespace aginfer::internal
