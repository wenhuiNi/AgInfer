#include "patch_projection_payload.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  aginfer::internal::PatchProjectionPayloadView parsed;
  auto status = aginfer::internal::ParsePatchProjectionPayload(
      bytes.data(), bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.grid != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.block != std::array<std::uint32_t, 3>{256, 1, 1} ||
      parsed.image_bytes != 602112 || parsed.weight_bytes != 2709504 ||
      parsed.bias_bytes != 4608 || parsed.output_bytes != 1179648 ||
      parsed.patch_workspace_bytes != 602112 || parsed.module_bytes != 160000 ||
      parsed.cublaslt.m != 256 || parsed.cublaslt.n != 1152 ||
      parsed.cublaslt.k != 588 || parsed.cublaslt.workspace_bytes != 2304 ||
      std::string(aginfer::internal::PatchProjectionSymbol()) !=
          "aginfer_patchify_f32_nchw_224_p14") {
    std::cerr << "valid patch projection payload did not parse\n";
    return 1;
  }
  for (std::size_t offset : {std::size_t{0}, std::size_t{24},
                             std::size_t{108}, std::size_t{192},
                             bytes.size() - 1}) {
    auto corrupt = bytes;
    corrupt[offset] ^= 1;
    if (aginfer::internal::ParsePatchProjectionPayload(
            corrupt.data(), corrupt.size(), &parsed).ok()) {
      std::cerr << "corrupt patch projection payload was accepted\n";
      return 1;
    }
  }
  if (aginfer::internal::ParsePatchProjectionPayload(
          bytes.data(), bytes.size() - 1, &parsed).ok()) {
    std::cerr << "truncated patch projection payload was accepted\n";
    return 1;
  }
  return 0;
}
