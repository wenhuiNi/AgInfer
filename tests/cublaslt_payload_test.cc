#include "cublaslt_payload.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

using aginfer::internal::CublasLtDType;
using aginfer::internal::CublasLtLinearPayloadView;
using aginfer::internal::ParseCublasLtLinearPayload;

void StoreU16(std::uint8_t* data, std::uint16_t value) {
  data[0] = static_cast<std::uint8_t>(value);
  data[1] = static_cast<std::uint8_t>(value >> 8);
}

void StoreU32(std::uint8_t* data, std::uint32_t value) {
  for (unsigned index = 0; index < 4; ++index) {
    data[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

void StoreU64(std::uint8_t* data, std::uint64_t value) {
  for (unsigned index = 0; index < 8; ++index) {
    data[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

bool Rejected(const std::vector<std::uint8_t>& bytes, std::size_t base = 0) {
  CublasLtLinearPayloadView parsed;
  return !ParseCublasLtLinearPayload(bytes.data() + base, bytes.size() - base,
                                     &parsed)
              .ok();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (!input.is_open() || bytes.size() != 192) return 3;

  std::vector<std::uint8_t> unaligned(bytes.size() + 1);
  std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
  CublasLtLinearPayloadView parsed;
  if (!ParseCublasLtLinearPayload(unaligned.data() + 1, bytes.size(), &parsed).ok()) {
    return 4;
  }
  if (parsed.target_arch != 120 || parsed.cublaslt_version != 120805 ||
      parsed.dtype != CublasLtDType::kBf16 || parsed.m != 50 ||
      parsed.n != 256 || parsed.k != 1024 || parsed.lda != 1024 ||
      parsed.ldb != 1024 || parsed.ldc != 256 || parsed.ldd != 256 ||
      parsed.workspace_bytes != 4 * 1024 * 1024 ||
      parsed.algorithm.algorithm_id != 23 || parsed.algorithm.tile_id != 20 ||
      parsed.algorithm.split_k != 1 || parsed.algorithm.stages_id != 14 ||
      parsed.x_alignment != 256 || parsed.weight_alignment != 256 ||
      parsed.bias_alignment != 256 || parsed.output_alignment != 256 ||
      parsed.workspace_alignment != 256) {
    return 5;
  }

  auto short_payload = bytes;
  short_payload.pop_back();
  if (!Rejected(short_payload)) return 6;
  auto bad_magic = bytes;
  bad_magic[0] = 0;
  if (!Rejected(bad_magic)) return 7;
  auto bad_schema = bytes;
  StoreU16(bad_schema.data() + 8, 2);
  if (!Rejected(bad_schema)) return 8;
  auto bad_arch = bytes;
  StoreU32(bad_arch.data() + 16, 121);
  if (!Rejected(bad_arch)) return 9;
  auto bad_dtype = bytes;
  StoreU32(bad_dtype.data() + 24, 99);
  if (!Rejected(bad_dtype)) return 10;
  auto bad_flags = bytes;
  StoreU32(bad_flags.data() + 36, 0);
  if (!Rejected(bad_flags)) return 11;
  auto bad_stride = bytes;
  StoreU64(bad_stride.data() + 92, 1);
  if (!Rejected(bad_stride)) return 12;
  auto bad_shape = bytes;
  StoreU64(bad_shape.data() + 68, 0);
  if (!Rejected(bad_shape)) return 13;
  auto bad_algorithm = bytes;
  StoreU32(bad_algorithm.data() + 132, 0xFFFFFFFFU);
  if (!Rejected(bad_algorithm)) return 14;
  auto bad_split = bytes;
  StoreU32(bad_split.data() + 140, 0xFFFFFFFFU);
  if (!Rejected(bad_split)) return 15;
  auto bad_alignment = bytes;
  StoreU32(bad_alignment.data() + 164, 3);
  if (!Rejected(bad_alignment)) return 16;
  auto bad_reserved = bytes;
  bad_reserved[184] = 1;
  if (!Rejected(bad_reserved)) return 17;
  auto tf32=bytes;
  StoreU16(tf32.data()+10,1);StoreU32(tf32.data()+28,2);
  if(!Rejected(tf32))return 18; // BF16 cannot select TF32.
  StoreU32(tf32.data()+24,1);
  if(!ParseCublasLtLinearPayload(tf32.data(),tf32.size(),&parsed).ok() || parsed.compute_mode!=2)return 19;
  StoreU16(tf32.data()+10,0);
  if(!Rejected(tf32))return 20;
  return 0;
}
