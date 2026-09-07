#include "prefix_input_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::ParsePrefixInputPayload;
using aginfer::internal::PrefixInputPayloadView;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  PrefixInputPayloadView parsed;
  const auto status = ParsePrefixInputPayload(bytes.data(), bytes.size(), &parsed);
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
  PrefixInputPayloadView parsed;
  auto status = ParsePrefixInputPayload(unaligned.data() + 1, bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.grid != std::array<std::uint32_t, 3>{7744, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.image_alignment != 4 || parsed.embedding_alignment != 2 ||
      parsed.token_alignment != 4 || parsed.mask_alignment != 1 ||
      parsed.prefix_alignment != 4 || parsed.position_alignment != 4 ||
      parsed.image_bytes != 2097152 || parsed.embedding_bytes != 1053294592 ||
      parsed.token_bytes != 800 || parsed.prefix_bytes != 7929856 ||
      parsed.module_bytes != 150000) {
    std::cerr << "valid unaligned prefix input payload failed\n";
    return 4;
  }
  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {20, "read count"},
           {44, "vocabulary"},
           {52, "grid"},
           {56, "block"},
           {92, "image bytes"},
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
  if (ParsePrefixInputPayload(bytes.data(), bytes.size(), nullptr).ok()) return 7;
  return 0;
}
