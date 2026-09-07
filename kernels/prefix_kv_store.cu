#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr std::uint64_t kVectors = 1ULL * 1 * 968 * 256 / 8;

}  // namespace

// With one KV head the row-major bytes of V in [1,968,1,256] and
// [1,1,968,256] order are identical. Both persistent states are stored in one
// launch without a separate layout kernel.
extern "C" __global__ void aginfer_prefix_kv_store_bf16_h1_s968_d256(
    const uint4* key, const uint4* value, uint4* state_key,
    uint4* state_value) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < 2 * kVectors) {
    const bool value_half = index >= kVectors;
    const std::uint64_t local = value_half ? index - kVectors : index;
    (value_half ? state_value : state_key)[local] =
        (value_half ? value : key)[local];
  }
}
