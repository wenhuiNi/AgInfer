#include "cuda_kernel_payload.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

using aginfer::internal::CudaKernelDType;
using aginfer::internal::CudaKernelId;
using aginfer::internal::CudaKernelPayloadView;
using aginfer::internal::CudaKernelSymbol;
using aginfer::internal::ParseCudaKernelPayload;

void StoreU32(std::uint8_t* data, std::uint32_t value) {
  for (unsigned index = 0; index < 4; ++index) {
    data[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

bool Rejected(const std::vector<std::uint8_t>& bytes, std::size_t base = 0) {
  CudaKernelPayloadView parsed;
  return !ParseCudaKernelPayload(bytes.data() + base, bytes.size() - base,
                                 &parsed)
              .ok();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (!input.is_open() || bytes.size() != 128) return 3;

  std::vector<std::uint8_t> unaligned(bytes.size() + 1);
  std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
  CudaKernelPayloadView parsed;
  if (!ParseCudaKernelPayload(unaligned.data() + 1, bytes.size(), &parsed).ok()) {
    return 4;
  }
  if (parsed.target_arch != 120 ||
      parsed.kernel_id != CudaKernelId::kCastBf16ToF32 ||
      parsed.input_dtype != CudaKernelDType::kBf16 ||
      parsed.output_dtype != CudaKernelDType::kF32 || parsed.numel != 51200 ||
      parsed.grid_x != 200 || parsed.block_x != 256 ||
      parsed.shared_bytes != 0 || parsed.module_bytes != 12296 ||
      parsed.input_alignment != 2 || parsed.output_alignment != 4 ||
      std::string(CudaKernelSymbol(parsed.kernel_id)) !=
          "aginfer_cast_bf16_to_f32") {
    return 5;
  }
  auto short_payload = bytes;
  short_payload.pop_back();
  if (!Rejected(short_payload)) return 6;
  auto bad_magic = bytes;
  bad_magic[0] = 0;
  if (!Rejected(bad_magic)) return 7;
  auto bad_arch = bytes;
  StoreU32(bad_arch.data() + 16, 121);
  if (!Rejected(bad_arch)) return 8;
  auto bad_kernel = bytes;
  StoreU32(bad_kernel.data() + 20, 99);
  if (!Rejected(bad_kernel)) return 9;
  auto bad_flags = bytes;
  StoreU32(bad_flags.data() + 32, 0);
  if (!Rejected(bad_flags)) return 10;
  auto bad_grid = bytes;
  StoreU32(bad_grid.data() + 36, 199);
  if (!Rejected(bad_grid)) return 11;
  auto bad_block = bytes;
  StoreU32(bad_block.data() + 40, 0);
  if (!Rejected(bad_block)) return 12;
  auto bad_digest = bytes;
  std::fill(bad_digest.begin() + 64, bad_digest.begin() + 96, 0);
  if (!Rejected(bad_digest)) return 13;
  auto bad_alignment = bytes;
  StoreU32(bad_alignment.data() + 96, 3);
  if (!Rejected(bad_alignment)) return 14;
  auto bad_reserved = bytes;
  bad_reserved[112] = 1;
  if (!Rejected(bad_reserved)) return 15;
  return 0;
}
