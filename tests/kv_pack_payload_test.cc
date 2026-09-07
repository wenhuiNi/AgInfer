#include "kv_pack_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::KvPackPayloadView;
using aginfer::internal::ParseKvPackPayload;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  KvPackPayloadView parsed;
  const auto status = ParseKvPackPayload(bytes.data(), bytes.size(), &parsed);
  if (status.ok()) {
    std::cerr << label << " unexpectedly passed\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() != 192) return 3;
  std::vector<std::uint8_t> unaligned(bytes.size() + 1);
  std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
  KvPackPayloadView parsed;
  auto status = ParseKvPackPayload(unaligned.data() + 1, bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.grid != std::array<std::uint32_t, 3>{255, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.alignment != 16 || parsed.prefix_bytes != 495616 ||
      parsed.current_k_bytes != 25600 || parsed.current_v_bytes != 25600 ||
      parsed.packed_k_bytes != 521216 || parsed.packed_v_bytes != 521216 ||
      parsed.module_bytes != 128000) {
    std::cerr << "valid unaligned KV pack payload failed\n";
    return 4;
  }
  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {20, "dtype"},
           {32, "current V layout"},
           {48, "prefix sequence"},
           {56, "total sequence"},
           {64, "grid"},
           {76, "block"},
           {92, "prefix bytes"},
           {172, "reserved"},
       }) {
    auto changed = bytes;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto zero_digest = bytes;
  std::fill(zero_digest.begin() + 140, zero_digest.begin() + 172, 0);
  if (!ExpectFailure(zero_digest, "zero module digest")) return 5;
  auto truncated = bytes;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseKvPackPayload(bytes.data(), bytes.size(), nullptr).ok()) return 7;
  return 0;
}
