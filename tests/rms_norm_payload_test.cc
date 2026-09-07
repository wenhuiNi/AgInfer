#include "rms_norm_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::ParseRmsNormPayload;
using aginfer::internal::RmsNormPayloadView;
using aginfer::internal::RmsNormVariant;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  RmsNormPayloadView parsed;
  const auto status = ParseRmsNormPayload(bytes.data(), bytes.size(), &parsed);
  if (status.ok()) {
    std::cerr << label << " unexpectedly passed\n";
    return false;
  }
  return true;
}

std::vector<std::uint8_t> Slice(const std::vector<std::uint8_t>& bytes,
                                std::size_t offset) {
  return {bytes.begin() + static_cast<std::ptrdiff_t>(offset),
          bytes.begin() + static_cast<std::ptrdiff_t>(offset + 192)};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() != 384) return 3;

  const auto f32 = Slice(bytes, 0);
  const auto bf16 = Slice(bytes, 192);
  std::vector<std::uint8_t> unaligned(f32.size() + 1);
  std::copy(f32.begin(), f32.end(), unaligned.begin() + 1);
  RmsNormPayloadView parsed;
  auto status = ParseRmsNormPayload(unaligned.data() + 1, f32.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 || parsed.rows != 50 ||
      parsed.width != 1024 ||
      parsed.variant != RmsNormVariant::kF32Rows50Width1024 ||
      parsed.grid != std::array<std::uint32_t, 3>{50, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.input_bytes != 204800 || parsed.weight_bytes != 4096 ||
      parsed.output_bytes != 204800 || parsed.module_bytes != 64000) {
    std::cerr << "valid unaligned F32 RMSNorm payload failed\n";
    return 4;
  }
  status = ParseRmsNormPayload(bf16.data(), bf16.size(), &parsed);
  if (!status.ok() || parsed.rows != 968 || parsed.width != 2048 ||
      parsed.variant != RmsNormVariant::kBf16Rows968Width2048 ||
      parsed.grid != std::array<std::uint32_t, 3>{968, 1, 1} ||
      parsed.input_bytes != 3964928 || parsed.weight_bytes != 8192 ||
      parsed.output_bytes != 3964928 || parsed.module_bytes != 64000) {
    std::cerr << "valid BF16 RMSNorm payload failed\n";
    return 4;
  }

  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {24, "weight dtype"},
           {40, "rows"},
           {44, "width"},
           {48, "grid"},
           {92, "epsilon"},
           {96, "tensor bytes"},
           {160, "reserved"},
       }) {
    auto changed = f32;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto zero_digest = f32;
  std::fill(zero_digest.begin() + 128, zero_digest.begin() + 160, 0);
  if (!ExpectFailure(zero_digest, "zero module digest")) return 5;
  auto truncated = f32;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseRmsNormPayload(f32.data(), f32.size(), nullptr).ok()) return 7;
  return 0;
}
