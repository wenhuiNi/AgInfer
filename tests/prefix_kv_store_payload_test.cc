#include "prefix_kv_store_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::ParsePrefixKvStorePayload;
using aginfer::internal::PrefixKvStorePayloadView;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  PrefixKvStorePayloadView parsed;
  const auto status =
      ParsePrefixKvStorePayload(bytes.data(), bytes.size(), &parsed);
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
  PrefixKvStorePayloadView parsed;
  auto status =
      ParsePrefixKvStorePayload(unaligned.data() + 1, bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.grid != std::array<std::uint32_t, 3>{242, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.alignment != 16 || parsed.key_bytes != 495616 ||
      parsed.value_bytes != 495616 || parsed.state_key_bytes != 495616 ||
      parsed.state_value_bytes != 495616 || parsed.module_bytes != 128000) {
    std::cerr << "valid unaligned prefix KV store payload failed\n";
    return 4;
  }
  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {20, "dtype"},
           {28, "value layout"},
           {48, "sequence"},
           {56, "grid"},
           {68, "block"},
           {80, "alignment"},
           {92, "key bytes"},
           {132, "reserved u64"},
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
  if (ParsePrefixKvStorePayload(bytes.data(), bytes.size(), nullptr).ok()) {
    return 7;
  }
  return 0;
}
