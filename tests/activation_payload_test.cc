#include "cuda_kernel_payload.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using aginfer::internal::CudaKernelDType;
using aginfer::internal::CudaKernelId;
using aginfer::internal::CudaKernelPayloadView;
using aginfer::internal::CudaKernelSymbol;
using aginfer::internal::ParseCudaKernelPayload;

bool ExpectFailure(const std::vector<std::uint8_t>& bytes,
                   const std::string& label) {
  CudaKernelPayloadView parsed;
  const auto status =
      ParseCudaKernelPayload(bytes.data(), bytes.size(), &parsed);
  if (status.ok()) {
    std::cerr << label << " unexpectedly passed\n";
    return false;
  }
  return true;
}

std::vector<std::uint8_t> Slice(const std::vector<std::uint8_t>& bytes,
                                std::size_t offset) {
  return {bytes.begin() + static_cast<std::ptrdiff_t>(offset),
          bytes.begin() + static_cast<std::ptrdiff_t>(offset + 128)};
}

void StoreU64(std::uint8_t* data, std::uint64_t value) {
  for (unsigned index = 0; index < 8; ++index) {
    data[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (bytes.size() != 512) return 3;

  const std::array<CudaKernelId, 4> ids{
      CudaKernelId::kGeluBf16,
      CudaKernelId::kGeluF32,
      CudaKernelId::kGeluBf16,
      CudaKernelId::kSiluF32,
  };
  const std::array<CudaKernelDType, 4> dtypes{
      CudaKernelDType::kBf16,
      CudaKernelDType::kF32,
      CudaKernelDType::kBf16,
      CudaKernelDType::kF32,
  };
  const std::array<std::uint64_t, 4> numels{
      204800, 1101824, 15859712, 1024};
  const std::array<std::uint32_t, 4> grids{800, 4304, 61952, 4};
  const std::array<std::string, 4> symbols{
      "aginfer_gelu_tanh_bf16",
      "aginfer_gelu_tanh_f32",
      "aginfer_gelu_tanh_bf16",
      "aginfer_silu_f32",
  };
  for (std::size_t index = 0; index < ids.size(); ++index) {
    const auto payload = Slice(bytes, index * 128);
    std::vector<std::uint8_t> unaligned(payload.size() + 1);
    std::copy(payload.begin(), payload.end(), unaligned.begin() + 1);
    CudaKernelPayloadView parsed;
    const auto status =
        ParseCudaKernelPayload(unaligned.data() + 1, payload.size(), &parsed);
    if (!status.ok() || parsed.target_arch != 120 ||
        parsed.kernel_id != ids[index] || parsed.input_dtype != dtypes[index] ||
        parsed.output_dtype != dtypes[index] || parsed.numel != numels[index] ||
        parsed.grid_x != grids[index] || parsed.block_x != 256 ||
        parsed.shared_bytes != 0 || parsed.module_bytes != 100000 ||
        parsed.launch_abi !=
            aginfer::internal::CudaKernelLaunchAbi::kUnaryPointersNumel ||
        std::string(CudaKernelSymbol(parsed.kernel_id)) != symbols[index]) {
      std::cerr << "valid GELU payload failed at index " << index << "\n";
      return 4;
    }
  }

  const auto first = Slice(bytes, 0);
  auto bad_numel = first;
  StoreU64(bad_numel.data() + 48, 204801);
  if (!ExpectFailure(bad_numel, "unknown GELU problem")) return 5;
  auto bad_dtype = first;
  bad_dtype[24] = static_cast<std::uint8_t>(CudaKernelDType::kF32);
  if (!ExpectFailure(bad_dtype, "GELU dtype mismatch")) return 5;
  auto bad_abi = first;
  bad_abi[108] = 2;
  if (!ExpectFailure(bad_abi, "GELU launch ABI")) return 5;
  auto truncated = first;
  truncated.pop_back();
  if (!ExpectFailure(truncated, "truncated")) return 6;
  if (ParseCudaKernelPayload(first.data(), first.size(), nullptr).ok()) return 7;
  return 0;
}
