#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr std::uint64_t kPrefixVectors = 1ULL * 1 * 968 * 256 / 8;
constexpr std::uint64_t kCurrentVectors = 1ULL * 1 * 50 * 256 / 8;
constexpr std::uint64_t kPackedVectors = kPrefixVectors + kCurrentVectors;

}  // namespace

// BF16 data is copied as 16-byte vectors. With one KV head the physical
// row-major bytes of current_v BSHD [1,50,1,256] already equal BHSD
// [1,1,50,256], so no standalone transpose is required.
extern "C" __global__ void aginfer_kv_pack_bf16_h1_s968_s50_d256(
    const uint4* prefix_k, const uint4* prefix_v, const uint4* current_k,
    const uint4* current_v, uint4* packed_k, uint4* packed_v) {
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < 2 * kPackedVectors; index += stride) {
    const bool value_half = index >= kPackedVectors;
    const std::uint64_t output_index =
        value_half ? index - kPackedVectors : index;
    const uint4* prefix = value_half ? prefix_v : prefix_k;
    const uint4* current = value_half ? current_v : current_k;
    uint4* output = value_half ? packed_v : packed_k;
    output[output_index] = output_index < kPrefixVectors
                               ? prefix[output_index]
                               : current[output_index - kPrefixVectors];
  }
}
