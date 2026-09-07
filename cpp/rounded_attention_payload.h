#pragma once
#include "status.h"
#include <array>
#include <cstddef>
#include <cstdint>
namespace aginfer::internal {
struct RoundedAttentionPayloadView {
  std::uint32_t arch=0,variant=0,query_length=0,key_length=0;
  std::uint32_t query_heads=0,kv_heads=0,head_dim=0;
  std::uint64_t module_bytes=0;
  std::array<std::uint8_t,32> module_sha256{};
  std::uint32_t cublaslt_version=0;
  std::uint64_t workspace_bytes=0;
  std::array<std::int32_t,9> qk_algorithm{},pv_algorithm{};
};
Status ParseRoundedAttentionPayload(const std::uint8_t* data,std::size_t size,RoundedAttentionPayloadView* output);
}
