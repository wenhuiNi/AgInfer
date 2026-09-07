#include "adaptive_rms_norm_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::AdaptiveRmsNormPayloadView;
using aginfer::internal::ParseAdaptiveRmsNormPayload;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  AdaptiveRmsNormPayloadView parsed;
  const auto status =
      ParseAdaptiveRmsNormPayload(bytes.data(), bytes.size(), &parsed);
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
  AdaptiveRmsNormPayloadView parsed;
  auto status = ParseAdaptiveRmsNormPayload(unaligned.data() + 1, bytes.size(),
                                            &parsed);
  if (!status.ok() || parsed.target_arch != 120 || parsed.rows != 50 ||
      parsed.width != 1024 || parsed.modulation_width != 3072 ||
      parsed.grid != std::array<std::uint32_t, 3>{50, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.hidden_bytes != 102400 || parsed.modulation_bytes != 12288 ||
      parsed.normalized_bytes != 102400 || parsed.gate_bytes != 102400 ||
      parsed.module_bytes != 96000) {
    std::cerr << "valid unaligned adaptive RMSNorm payload failed\n";
    return 4;
  }

  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {20, "input dtype"},
           {44, "rows"},
           {48, "width"},
           {52, "modulation width"},
           {60, "grid"},
           {72, "block"},
           {108, "epsilon"},
           {112, "tensor bytes"},
           {184, "reserved"},
       }) {
    auto changed = bytes;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto zero_digest = bytes;
  std::fill(zero_digest.begin() + 152, zero_digest.begin() + 184, 0);
  if (!ExpectFailure(zero_digest, "zero module digest")) return 5;
  auto truncated = bytes;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseAdaptiveRmsNormPayload(bytes.data(), bytes.size(), nullptr).ok()) {
    return 7;
  }
  return 0;
}
