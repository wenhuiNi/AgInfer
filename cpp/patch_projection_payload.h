#pragma once

#include "cublaslt_payload.h"
#include "status.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kPatchProjectionPayloadSize = 384;

struct PatchProjectionPayloadView {
  std::uint32_t target_arch = 0;
  std::array<std::uint32_t, 3> grid{0, 0, 0};
  std::array<std::uint32_t, 3> block{0, 0, 0};
  std::uint32_t image_alignment = 0;
  std::uint32_t weight_alignment = 0;
  std::uint32_t bias_alignment = 0;
  std::uint32_t output_alignment = 0;
  std::uint32_t workspace_alignment = 0;
  std::uint64_t image_bytes = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t bias_bytes = 0;
  std::uint64_t output_bytes = 0;
  std::uint64_t patch_workspace_bytes = 0;
  std::uint64_t module_bytes = 0;
  std::array<std::uint8_t, 32> module_sha256{};
  CublasLtLinearPayloadView cublaslt;
};

Status ParsePatchProjectionPayload(const std::uint8_t* data, std::size_t size,
                                   PatchProjectionPayloadView* output);
const char* PatchProjectionSymbol() noexcept;

}  // namespace aginfer::internal
