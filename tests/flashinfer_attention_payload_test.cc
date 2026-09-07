#include "flashinfer_attention_payload.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::FlashInferAttentionPayloadView;
using aginfer::internal::ParseFlashInferAttentionPayload;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  FlashInferAttentionPayloadView parsed;
  const auto status =
      ParseFlashInferAttentionPayload(bytes.data(), bytes.size(), &parsed);
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
  FlashInferAttentionPayloadView parsed;
  auto status = ParseFlashInferAttentionPayload(unaligned.data() + 1, bytes.size(),
                                                &parsed);
  if (!status.ok() || parsed.target_arch != 120 || parsed.num_q_heads != 8 ||
      parsed.num_kv_heads != 1 || parsed.query_length != 50 ||
      parsed.key_length != 1018 || parsed.head_dim != 256 ||
      parsed.grid_x != 7 || parsed.block_y != 4 ||
      parsed.shared_bytes != 49152 || parsed.mask_bytes != 50900 ||
      parsed.output_bytes != 204800) {
    std::cerr << "valid unaligned payload failed\n";
    return 4;
  }

  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {52, "query length"},
           {92, "scale"},
           {136, "FlashInfer commit"},
           {156, "CCCL commit"},
           {176, "reserved"},
       }) {
    auto changed = bytes;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto truncated = bytes;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseFlashInferAttentionPayload(bytes.data(), bytes.size(), nullptr).ok()) {
    std::cerr << "null output unexpectedly passed\n";
    return 7;
  }
  return 0;
}
