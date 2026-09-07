#include "patch_projection_payload.h"

#include <algorithm>
#include <array>
#include <string>

namespace aginfer::internal {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    'A', 'I', 'P', 'P', 'A', '1', 0, 0};

Status Error(const std::string& message) {
  return Status(StatusCode::kCorruptPackage, message);
}

std::uint16_t U16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) |
         static_cast<std::uint16_t>(p[1]) << 8;
}
std::uint32_t U32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         static_cast<std::uint32_t>(p[1]) << 8 |
         static_cast<std::uint32_t>(p[2]) << 16 |
         static_cast<std::uint32_t>(p[3]) << 24;
}
std::uint64_t U64(const std::uint8_t* p) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(p[index]) << (index * 8);
  }
  return value;
}

}  // namespace

const char* PatchProjectionSymbol() noexcept {
  return "aginfer_patchify_f32_nchw_224_p14";
}

Status ParsePatchProjectionPayload(const std::uint8_t* data, std::size_t size,
                                   PatchProjectionPayloadView* output) {
  if (data == nullptr || output == nullptr ||
      size != kPatchProjectionPayloadSize) {
    return Error("patch projection payload must have its fixed size");
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), data) || U16(data + 8) != 1 ||
      U16(data + 10) != 0 || U32(data + 12) != 120 ||
      U32(data + 16) != 1 || U32(data + 20) != 1 ||
      U32(data + 24) != 256 || U32(data + 28) != 1 ||
      U32(data + 32) != 1 || U32(data + 36) != 256 ||
      U32(data + 40) != 1 || U32(data + 44) != 1 ||
      U32(data + 48) != 4 || U32(data + 52) != 4 ||
      U32(data + 56) != 4 || U32(data + 60) != 4 ||
      U32(data + 64) != 256 || U32(data + 68) != 1 ||
      U32(data + 72) != 3 || U32(data + 76) != 224 ||
      U32(data + 80) != 224 || U32(data + 84) != 14 ||
      U32(data + 88) != 14 || U32(data + 92) != 1152 ||
      U32(data + 96) != 16 || U32(data + 100) != 16 ||
      U32(data + 104) != 0 || U64(data + 108) != 602112 ||
      U64(data + 116) != 2709504 || U64(data + 124) != 4608 ||
      U64(data + 132) != 1179648 || U64(data + 140) != 602112 ||
      U64(data + 148) == 0 ||
      !std::any_of(data + 156, data + 188,
                   [](std::uint8_t value) { return value != 0; }) ||
      !std::all_of(data + 188, data + 192,
                   [](std::uint8_t value) { return value == 0; })) {
    return Error("patch projection payload is outside the delivered exact variant");
  }
  CublasLtLinearPayloadView cublaslt;
  Status status = ParseCublasLtLinearPayload(data + 192, 192, &cublaslt);
  if (!status.ok()) return status;
  if (cublaslt.target_arch != 120 || cublaslt.dtype != CublasLtDType::kF32 ||
      cublaslt.m != 256 || cublaslt.n != 1152 || cublaslt.k != 588) {
    return Error("patch projection contains the wrong cuBLASLt problem");
  }
  PatchProjectionPayloadView parsed;
  parsed.target_arch = 120;
  parsed.grid = {256, 1, 1};
  parsed.block = {256, 1, 1};
  parsed.image_alignment = 4;
  parsed.weight_alignment = 4;
  parsed.bias_alignment = 4;
  parsed.output_alignment = 4;
  parsed.workspace_alignment = 256;
  parsed.image_bytes = 602112;
  parsed.weight_bytes = 2709504;
  parsed.bias_bytes = 4608;
  parsed.output_bytes = 1179648;
  parsed.patch_workspace_bytes = 602112;
  parsed.module_bytes = U64(data + 148);
  std::copy(data + 156, data + 188, parsed.module_sha256.begin());
  parsed.cublaslt = cublaslt;
  *output = parsed;
  return Status::Ok();
}

}  // namespace aginfer::internal
