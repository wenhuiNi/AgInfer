#include "action_slice_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {
using aginfer::internal::ActionSlicePayloadView;
using aginfer::internal::ParseActionSlicePayload;
bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  ActionSlicePayloadView parsed;
  const auto status = ParseActionSlicePayload(bytes.data(), bytes.size(), &parsed);
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
  ActionSlicePayloadView parsed;
  auto status = ParseActionSlicePayload(unaligned.data() + 1, bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.grid != std::array<std::uint32_t, 3>{2, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.input_alignment != 4 || parsed.output_alignment != 4 ||
      parsed.input_bytes != 6400 || parsed.output_bytes != 1400 ||
      parsed.module_bytes != 181000) {
    std::cerr << "valid unaligned action slice payload failed\n";
    return 4;
  }
  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {20, "dtype"},
           {28, "rows"},
           {32, "input width"},
           {36, "output width"},
           {44, "grid"},
           {48, "block"},
           {68, "axis"},
           {76, "stop"},
           {92, "input bytes"},
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
  if (ParseActionSlicePayload(bytes.data(), bytes.size(), nullptr).ok()) return 7;
  return 0;
}
