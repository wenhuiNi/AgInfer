#include "gelu_mul_payload.h"
#include <algorithm>
#include <cstring>

namespace aginfer::internal {
namespace {
std::uint64_t Read(const std::uint8_t* p, unsigned count) {
  std::uint64_t result = 0;
  for (unsigned i = 0; i < count; ++i) result |= std::uint64_t(p[i]) << (8 * i);
  return result;
}
}
Status ParseGeluMulPayload(const std::uint8_t* data, std::size_t size, GeluMulPayloadView* out) {
  auto bad = [] { return Status(StatusCode::kCorruptPackage, "invalid GELU-multiply payload"); };
  if (!data || !out || size != 128) return bad();
  const bool tiled = std::memcmp(data, "AIGMT1\0", 8) == 0;
  const bool packed = tiled || std::memcmp(data, "AIGMP1\0", 8) == 0;
  if ((!packed && std::memcmp(data, "AIGMU1\0", 8)) || Read(data + 8, 4) != 1
      || Read(data + 12, 4) != 120 || !Read(data + 16, 8) || Read(data + 16, 8) > (1U << 26)
      || !Read(data + 24, 8) || !std::any_of(data + 32, data + 64, [](auto x) { return x != 0; })
      || !std::all_of(data + 68, data + 128, [](auto x) { return x == 0; })) return bad();
  GeluMulPayloadView p;
  p.numel = Read(data + 16, 8); p.module_bytes = Read(data + 24, 8);
  p.packed_width = Read(data + 64, 4);
  p.packed_tiled = tiled;
  if (packed ? (!p.packed_width || p.packed_width > 32768 || p.numel % p.packed_width ||
                p.numel / p.packed_width > 2048) : p.packed_width != 0) return bad();
  std::copy_n(data + 32, 32, p.module_sha256.begin());
  *out = p;
  return Status::Ok();
}
}  // namespace aginfer::internal
