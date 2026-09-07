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
  std::array<std::uint32_t, 3> grid{}, block{256, 1, 1};
  Status Execute(CudaStream stream) override {
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
  if (module.arch != 120 || module.bytes != p.module_bytes || module.sha256 != p.module_sha256
      || !module.driver || !module.module || b.size() != 3) return bad();
  const auto bytes = p.numel * 2;
  for (int i = 0; i < 3; ++i) {
    if (!b[i].data || reinterpret_cast<std::uintptr_t>(b[i].data) % 2 || b[i].bytes < bytes
        || b[i].access != (i == 2 ? CommandOperandAccess::kWrite : CommandOperandAccess::kRead)) return bad();
  }
  // Read-only inputs may alias; the output may overlap neither input.
  auto output = reinterpret_cast<std::uintptr_t>(b[2].data);
  for (int i = 0; i < 2; ++i) {
    auto input = reinterpret_cast<std::uintptr_t>(b[i].data);
    if (output <= input ? input - output < bytes : output - input < bytes) return bad();
  }
  auto function = module.driver->GetFunction(module.module, "aginfer_gelu_tanh_mul_bf16");
  if (!function.ok()) return function.status();
  auto command = std::make_unique<GeluMul>();
  command->driver = module.driver; command->function = function.value(); command->numel = p.numel;
  for (int i = 0; i < 3; ++i) command->pointers[i] = b[i].data;
  command->grid = {static_cast<std::uint32_t>(std::min<std::uint64_t>((p.numel + 255) / 256, 4096)), 1, 1};
  *out = std::move(command);
  return Status::Ok();
}
}  // namespace aginfer::internal
