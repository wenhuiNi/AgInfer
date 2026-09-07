#include "layer_norm_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::LayerNormPayloadView;
using aginfer::internal::ParseLayerNormPayload;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  LayerNormPayloadView parsed;
  const auto status = ParseLayerNormPayload(bytes.data(), bytes.size(), &parsed);
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
  LayerNormPayloadView parsed;
  auto status = ParseLayerNormPayload(unaligned.data() + 1, bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 || parsed.rows != 256 ||
      parsed.width != 1152 ||
      parsed.grid != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.input_bytes != 1179648 || parsed.weight_bytes != 4608 ||
      parsed.bias_bytes != 4608 || parsed.output_bytes != 1179648 ||
      parsed.module_bytes != 64000) {
    std::cerr << "valid unaligned LayerNorm payload failed\n";
    return 4;
  }

  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {32, "rows"},
           {36, "width"},
           {40, "grid"},
           {88, "epsilon"},
           {92, "tensor bytes"},
           {164, "reserved"},
       }) {
    auto changed = bytes;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto zero_digest = bytes;
  std::fill(zero_digest.begin() + 132, zero_digest.begin() + 164, 0);
  if (!ExpectFailure(zero_digest, "zero module digest")) return 5;
  auto truncated = bytes;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseLayerNormPayload(bytes.data(), bytes.size(), nullptr).ok()) return 7;
  return 0;
}
