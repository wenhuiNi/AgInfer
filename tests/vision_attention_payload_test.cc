#include "vision_attention_payload.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::ParseVisionAttentionPayload;
using aginfer::internal::VisionAttentionPayloadView;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  VisionAttentionPayloadView parsed;
  const auto status =
      ParseVisionAttentionPayload(bytes.data(), bytes.size(), &parsed);
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
  VisionAttentionPayloadView parsed;
  auto status = ParseVisionAttentionPayload(unaligned.data() + 1, bytes.size(),
                                            &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.query_length != 256 || parsed.key_length != 256 ||
      parsed.num_heads != 16 || parsed.head_dim != 72 ||
      parsed.grid != std::array<std::uint32_t, 3>{256, 16, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.shared_bytes != 0 || parsed.query_bytes != 1179648 ||
      parsed.mask_bytes != 1 || parsed.output_bytes != 1179648 ||
      parsed.module_bytes != 48608) {
    std::cerr << "valid unaligned vision attention payload failed\n";
    return 4;
  }

  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {48, "query length"},
           {64, "grid"},
           {96, "scale"},
           {100, "tensor bytes"},
           {180, "reserved"},
       }) {
    auto changed = bytes;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto zero_digest = bytes;
  std::fill(zero_digest.begin() + 148, zero_digest.begin() + 180, 0);
  if (!ExpectFailure(zero_digest, "zero module digest")) return 5;
  auto truncated = bytes;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseVisionAttentionPayload(bytes.data(), bytes.size(), nullptr).ok()) {
    std::cerr << "null output unexpectedly passed\n";
    return 7;
  }
  return 0;
}
