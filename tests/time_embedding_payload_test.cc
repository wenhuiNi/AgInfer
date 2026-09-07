#include "time_embedding_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::ParseTimeEmbeddingPayload;
using aginfer::internal::TimeEmbeddingPayloadView;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  TimeEmbeddingPayloadView parsed;
  const auto status = ParseTimeEmbeddingPayload(bytes.data(), bytes.size(), &parsed);
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
  TimeEmbeddingPayloadView parsed;
  auto status = ParseTimeEmbeddingPayload(unaligned.data() + 1, bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.grid != std::array<std::uint32_t, 3>{2, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.input_alignment != 4 || parsed.output_alignment != 4 ||
      parsed.input_bytes != 4 || parsed.output_bytes != 4096 ||
      parsed.module_bytes != 153000) {
    std::cerr << "valid unaligned time embedding payload failed\n";
    return 4;
  }
  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {20, "input dtype"},
           {28, "dimension"},
           {36, "grid"},
           {40, "block"},
           {44, "alignment"},
           {92, "input bytes"},
           {116, "minimum period"},
           {124, "maximum period"},
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
  if (ParseTimeEmbeddingPayload(bytes.data(), bytes.size(), nullptr).ok()) return 7;
  return 0;
}
