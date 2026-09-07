#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kPatchInputChannels = 3;
constexpr int kPatchInputHeight = 224;
constexpr int kPatchInputWidth = 224;
constexpr int kPatchHeight = 14;
constexpr int kPatchWidth = 14;
constexpr int kPatchGrid = 16;
constexpr int kPatchElements =
    kPatchInputChannels * kPatchHeight * kPatchWidth;

}  // namespace

// Convert one NCHW image into the row-major [256,588] matrix consumed by the
// exact cuBLASLt projection. Patches do not overlap because stride == extent.
extern "C" __global__ void aginfer_patchify_f32_nchw_224_p14(
    const float* image, float* patches) {
  const int patch = static_cast<int>(blockIdx.x);
  if (patch >= kPatchGrid * kPatchGrid) return;
  const int patch_y = patch / kPatchGrid;
  const int patch_x = patch % kPatchGrid;
  for (int element = static_cast<int>(threadIdx.x); element < kPatchElements;
       element += static_cast<int>(blockDim.x)) {
    const int channel = element / (kPatchHeight * kPatchWidth);
    const int offset = element % (kPatchHeight * kPatchWidth);
    const int y = patch_y * kPatchHeight + offset / kPatchWidth;
    const int x = patch_x * kPatchWidth + offset % kPatchWidth;
    const std::uint64_t source =
        (static_cast<std::uint64_t>(channel) * kPatchInputHeight + y) *
            kPatchInputWidth +
        x;
    patches[static_cast<std::uint64_t>(patch) * kPatchElements + element] =
        image[source];
  }
}
