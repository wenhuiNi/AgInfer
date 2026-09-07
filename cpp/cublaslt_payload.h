#pragma once

#include "status.h"

#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

inline constexpr std::size_t kCublasLtLinearPayloadSize = 192;
inline constexpr std::uint16_t kCublasLtLinearSchemaMajor = 1;
inline constexpr std::uint16_t kCublasLtLinearSchemaMinor = 1;

enum class CublasLtDType : std::uint32_t { kF32 = 1, kBf16 = 2 };

struct CublasLtAlgorithmConfig {
  std::int32_t algorithm_id = 0;
  std::uint32_t tile_id = 0;
  std::int32_t split_k = 1;
  std::uint32_t reduction_scheme = 0;
  std::uint32_t cta_swizzling = 0;
  std::uint32_t custom_option = 0;
  std::uint32_t stages_id = 0;
  std::uint16_t inner_shape_id = 0;
  std::uint16_t cluster_shape_id = 0;
};

struct CublasLtLinearPayloadView {
  std::uint32_t target_arch = 0;
  std::uint32_t cublaslt_version = 0;
  CublasLtDType dtype = CublasLtDType::kF32;
  std::uint32_t compute_mode = 1;
  std::uint64_t m = 0;
  std::uint64_t n = 0;
  std::uint64_t k = 0;
  std::uint64_t lda = 0;
  std::uint64_t ldb = 0;
  std::uint64_t ldc = 0;
  std::uint64_t ldd = 0;
  std::uint64_t workspace_bytes = 0;
  CublasLtAlgorithmConfig algorithm;
  std::uint32_t x_alignment = 0;
  std::uint32_t weight_alignment = 0;
  std::uint32_t bias_alignment = 0;
  std::uint32_t output_alignment = 0;
  std::uint32_t workspace_alignment = 0;
};

Status ParseCublasLtLinearPayload(const std::uint8_t* data, std::size_t size,
                                  CublasLtLinearPayloadView* output);

}  // namespace aginfer::internal
