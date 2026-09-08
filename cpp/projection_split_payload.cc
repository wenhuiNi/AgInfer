#include "projection_split_payload.h"
#include <algorithm>
#include <cstring>

namespace aginfer::internal {
namespace {
std::uint32_t Read32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) | std::uint32_t(p[1]) << 8 | std::uint32_t(p[2]) << 16 | std::uint32_t(p[3]) << 24;
}
}
Status ParseProjectionSplitPayload(const std::uint8_t* data, std::size_t size, ProjectionSplitPayloadView* out) {
  auto bad = [] { return Status(StatusCode::kCorruptPackage, "invalid projection split payload"); };
  if (!data || !out || size != 128
      || (std::memcmp(data, "AIPSP1\0", 8) && std::memcmp(data, "AIQRP1\0", 8)) || Read32(data + 8) != 1
      || Read32(data + 12) != 120 || !Read32(data + 16) || Read32(data + 16) > 128
      || !std::all_of(data + 72, data + 128, [](auto x) { return x == 0; })
      || !std::any_of(data + 40, data + 72, [](auto x) { return x != 0; })) return bad();
  ProjectionSplitPayloadView p;
  p.rope_pack = !std::memcmp(data, "AIQRP1\0", 8);
  p.rows = Read32(data + 16);
  std::uint64_t sum = 0;
  for (int i = 0; i < 3; ++i) {
    p.widths[i] = Read32(data + 20 + 4 * i);
    if (!p.widths[i] || p.widths[i] % 8) return bad();
    sum += p.widths[i];
  }
  if (sum > 65536) return bad();
  if (p.rope_pack && (p.rows != 50 || p.widths != std::array<std::uint32_t, 3>{2048, 256, 256})) return bad();
  p.module_bytes = Read32(data + 32) | (std::uint64_t(Read32(data + 36)) << 32);
  if (!p.module_bytes) return bad();
  std::copy_n(data + 40, 32, p.module_sha256.begin());
  *out = p;
  return Status::Ok();
}
}  // namespace aginfer::internal
