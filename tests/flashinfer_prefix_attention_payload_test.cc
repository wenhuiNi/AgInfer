#include "flashinfer_attention_payload.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() != aginfer::internal::kFlashInferAttentionPayloadSize) return 3;

  aginfer::internal::FlashInferAttentionPayloadView parsed;
  auto status = aginfer::internal::ParseFlashInferAttentionPayload(
      bytes.data(), bytes.size(), &parsed);
  if (!status.ok() || parsed.target_arch != 120 ||
      parsed.variant !=
          aginfer::internal::FlashInferAttentionVariant::kBf16GqaPrefixPadBoolToBshd ||
      parsed.num_q_heads != 8 || parsed.num_kv_heads != 1 ||
      parsed.query_length != 968 || parsed.key_length != 968 ||
      parsed.head_dim != 256 || parsed.grid_x != 121 || parsed.block_y != 4 ||
      parsed.shared_bytes != 49152 || parsed.query_bytes != 3964928 ||
      parsed.key_bytes != 495616 || parsed.value_bytes != 495616 ||
      parsed.mask_bytes != 968 || parsed.output_bytes != 3964928) {
    std::cerr << "valid prefix payload failed\n";
    return 4;
  }

  auto wrong_mask_layout = bytes;
  wrong_mask_layout[36] = 1;
  if (aginfer::internal::ParseFlashInferAttentionPayload(
          wrong_mask_layout.data(), wrong_mask_layout.size(), &parsed)
          .ok()) {
    std::cerr << "wrong prefix mask layout unexpectedly passed\n";
    return 5;
  }
  auto truncated = bytes;
  truncated.pop_back();
  if (aginfer::internal::ParseFlashInferAttentionPayload(
          truncated.data(), truncated.size(), &parsed)
          .ok()) {
    std::cerr << "truncated prefix payload unexpectedly passed\n";
    return 6;
  }
  return 0;
}
