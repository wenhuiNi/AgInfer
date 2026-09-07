#include "rope_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::ParseRopePayload;
using aginfer::internal::RopePayloadView;
using aginfer::internal::RopeVariant;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  RopePayloadView parsed;
  const auto status = ParseRopePayload(bytes.data(), bytes.size(), &parsed);
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
  if (bytes.size() != 768) return 3;

  const std::array<RopeVariant, 4> variants{
      RopeVariant::kBf16Sequence968Heads8,
      RopeVariant::kBf16Sequence968Heads1,
      RopeVariant::kBf16Sequence50Heads8,
      RopeVariant::kBf16Sequence50Heads1,
  };
  const std::array<std::uint32_t, 4> sequences{968, 968, 50, 50};
  const std::array<std::uint32_t, 4> heads{8, 1, 8, 1};
  const std::array<std::uint32_t, 4> grids{3872, 484, 200, 25};
  const std::array<std::uint64_t, 4> tensor_bytes{
      3964928, 495616, 204800, 25600};
  const std::array<std::uint64_t, 4> position_bytes{3872, 3872, 200, 200};
  for (std::size_t index = 0; index < variants.size(); ++index) {
    auto payload = Slice(bytes, index * 192);
    std::vector<std::uint8_t> unaligned(payload.size() + 1);
    std::copy(payload.begin(), payload.end(), unaligned.begin() + 1);
    RopePayloadView parsed;
    const auto status =
        ParseRopePayload(unaligned.data() + 1, payload.size(), &parsed);
    if (!status.ok() || parsed.target_arch != 120 ||
        parsed.variant != variants[index] || parsed.sequence != sequences[index] ||
        parsed.heads != heads[index] || parsed.head_dim != 256 ||
        parsed.grid != std::array<std::uint32_t, 3>{grids[index], 1, 1} ||
        parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
        parsed.input_bytes != tensor_bytes[index] ||
        parsed.position_bytes != position_bytes[index] ||
        parsed.output_bytes != tensor_bytes[index] ||
        parsed.module_bytes != 80000) {
      std::cerr << "valid unaligned RoPE payload failed at variant " << index
                << "\n";
      return 4;
    }
  }

  const auto first = Slice(bytes, 0);
  for (const auto& mutation : {
           std::pair<std::size_t, std::string>{0, "magic"},
           {12, "architecture"},
           {16, "variant"},
           {24, "position dtype"},
           {32, "input layout"},
           {44, "heads"},
           {48, "sequence"},
           {60, "grid"},
           {100, "frequency dtype"},
           {104, "pairing"},
           {108, "theta"},
           {112, "tensor bytes"},
           {176, "reserved"},
       }) {
    auto changed = first;
    changed[mutation.first] ^= 1;
    if (!ExpectFailure(changed, mutation.second)) return 5;
  }
  auto zero_digest = first;
  std::fill(zero_digest.begin() + 144, zero_digest.begin() + 176, 0);
  if (!ExpectFailure(zero_digest, "zero module digest")) return 5;
  auto truncated = first;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseRopePayload(first.data(), first.size(), nullptr).ok()) return 7;
  return 0;
}
