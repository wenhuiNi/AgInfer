#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kPrefixImageTokens = 256;
constexpr int kPrefixImageCount = 3;
constexpr int kPrefixLanguageTokens = 200;
constexpr int kPrefixWidth = 2048;
constexpr int kPrefixVocabulary = 257152;
constexpr int kPrefixTotalTokens =
    kPrefixImageCount * kPrefixImageTokens + kPrefixLanguageTokens;

}  // namespace

extern "C" __global__ void aginfer_prefix_input_f32_bf16_s968_d2048(
    const float* image0, const float* image1, const float* image2,
    const __nv_bfloat16* embedding, const std::int32_t* tokens,
    const std::uint8_t* image_mask0, const std::uint8_t* image_mask1,
    const std::uint8_t* image_mask2, const std::uint8_t* token_mask,
    float* prefix, std::uint8_t* pad_mask, std::int32_t* positions) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  constexpr std::uint64_t kElements =
      static_cast<std::uint64_t>(kPrefixTotalTokens) * kPrefixWidth;
  if (index < kElements) {
    const int token = static_cast<int>(index / kPrefixWidth);
    const int channel = static_cast<int>(index % kPrefixWidth);
    if (token < kPrefixImageCount * kPrefixImageTokens) {
      const int image = token / kPrefixImageTokens;
      const std::uint64_t image_index =
          static_cast<std::uint64_t>(token % kPrefixImageTokens) *
              kPrefixWidth +
          channel;
      const float* source = image == 0 ? image0 : (image == 1 ? image1 : image2);
      prefix[index] = source[image_index];
    } else {
      const int language_token =
          token - kPrefixImageCount * kPrefixImageTokens;
      const int token_id = tokens[language_token];
      if (token_id < 0 || token_id >= kPrefixVocabulary) {
        prefix[index] = __int_as_float(0x7fc00000);
      } else {
        const __nv_bfloat16 value =
            embedding[static_cast<std::uint64_t>(token_id) * kPrefixWidth +
                      channel];
        const __nv_bfloat16 scale = __ushort_as_bfloat16(0x4235U);
        const __nv_bfloat16 scaled = __float2bfloat16_rn(
            __bfloat162float(value) * __bfloat162float(scale));
        prefix[index] = __bfloat162float(scaled);
      }
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int cumulative = 0;
    for (int token = 0; token < kPrefixTotalTokens; ++token) {
      std::uint8_t valid = 0;
      if (token < kPrefixImageTokens) {
        valid = image_mask0[0] != 0;
      } else if (token < 2 * kPrefixImageTokens) {
        valid = image_mask1[0] != 0;
      } else if (token < 3 * kPrefixImageTokens) {
        valid = image_mask2[0] != 0;
      } else {
        valid = token_mask[token - 3 * kPrefixImageTokens] != 0;
      }
      pad_mask[token] = valid;
      cumulative += valid != 0;
      positions[token] = cumulative - 1;
    }
  }
}
