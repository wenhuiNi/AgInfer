#include "providers/flashinfer_attention.h"

#include "flashinfer_attention_payload.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace aginfer::internal {
namespace {

using Element = __nv_bfloat16;
using Params = flashinfer::SinglePrefillParams<Element, Element, Element>;

// FlashInfer deliberately supports custom attention variants. Reading the
// ProgramIR's pre-broadcast 2-D BOOL mask here avoids a mask materialization or
// bit-packing launch and preserves arbitrary dense mask bits.
struct DenseBoolAttention : flashinfer::AttentionVariantBase {
  static constexpr bool use_softmax = true;

  const std::uint8_t* mask;
  std::uint32_t query_length;
  std::uint32_t key_length;
  std::uint32_t window_left;
  float sm_scale_log2;

  template <typename T>
  __device__ __host__ DenseBoolAttention(const T& params, std::uint32_t,
                                         std::uint8_t*)
      : mask(params.maybe_custom_mask),
        query_length(params.qo_len),
        key_length(params.kv_len),
        window_left(params.kv_len),
        sm_scale_log2(params.sm_scale * flashinfer::math::log2e) {}

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx,
                            qo_head_idx, kv_head_idx, { return logits; })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx,
                       kv_head_idx, {
    return qo_idx < query_length && kv_idx < key_length &&
           mask[static_cast<std::uint64_t>(qo_idx) * key_length + kv_idx] != 0;
  })
};

// ProgramIR applies a common finite value to an entirely masked query row, so
// softmax shift-invariance lets the kernel use an all-zero row and produce the
// same mean(V) result. Valid query rows still receive the finite BF16 minimum
// for masked keys, avoiding FlashInfer's hard-mask zero-row behavior.
struct PrefixPadBoolAttention : flashinfer::AttentionVariantBase {
  static constexpr bool use_softmax = true;

  const std::uint8_t* pad_mask;
  std::uint32_t query_length;
  std::uint32_t key_length;
  std::uint32_t window_left;
  float sm_scale_log2;

  template <typename T>
  __device__ __host__ PrefixPadBoolAttention(const T& params, std::uint32_t,
                                              std::uint8_t*)
      : pad_mask(params.maybe_custom_mask),
        query_length(params.qo_len),
        key_length(params.kv_len),
        window_left(params.kv_len),
        sm_scale_log2(params.sm_scale * flashinfer::math::log2e) {}

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx,
                            qo_head_idx, kv_head_idx, {
    if (qo_idx >= query_length || kv_idx >= key_length) return logits;
    if (pad_mask[qo_idx] == 0) return static_cast<T>(0.0F);
    return pad_mask[kv_idx] != 0 ? logits
                                 : static_cast<T>(-3.38953139e38F);
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx,
                       kv_head_idx, {
    return qo_idx < query_length && kv_idx < key_length;
  })
};

using Traits = flashinfer::KernelTraits<
    flashinfer::MaskMode::kCustom,
    /*CTA_TILE_Q=*/64,
    /*NUM_MMA_Q=*/1,
    /*NUM_MMA_KV=*/1,
    /*NUM_MMA_D_QK=*/16,
    /*NUM_MMA_D_VO=*/16,
    /*NUM_WARPS_Q=*/4,
    /*NUM_WARPS_KV=*/1,
    flashinfer::PosEncodingMode::kNone, Element, Element, Element, float,
    std::int32_t, DenseBoolAttention>;

using PrefixTraits = flashinfer::KernelTraits<
    flashinfer::MaskMode::kCustom,
    /*CTA_TILE_Q=*/64,
    /*NUM_MMA_Q=*/1,
    /*NUM_MMA_KV=*/1,
    /*NUM_MMA_D_QK=*/16,
    /*NUM_MMA_D_VO=*/16,
    /*NUM_WARPS_Q=*/4,
    /*NUM_WARPS_KV=*/1,
    flashinfer::PosEncodingMode::kNone, Element, Element, Element, float,
    std::int32_t, PrefixPadBoolAttention>;

static_assert(!Traits::IsInvalid());
static_assert(Traits::NUM_THREADS == 128);
static_assert(sizeof(typename Traits::SharedStorage) == 49152);
static_assert(!PrefixTraits::IsInvalid());
static_assert(PrefixTraits::NUM_THREADS == 128);
static_assert(sizeof(typename PrefixTraits::SharedStorage) == 49152);

__global__ __launch_bounds__(Traits::NUM_THREADS) void AgInferFlashInferAttentionKernel(
    const __grid_constant__ Params params) {
  extern __shared__ std::uint8_t shared[];
  auto& storage = *reinterpret_cast<typename Traits::SharedStorage*>(shared);
  flashinfer::SinglePrefillWithKVCacheDevice<Traits>(params, storage);
}

__global__ __launch_bounds__(PrefixTraits::NUM_THREADS)
void AgInferFlashInferPrefixAttentionKernel(const __grid_constant__ Params params) {
  extern __shared__ std::uint8_t shared[];
  auto& storage =
      *reinterpret_cast<typename PrefixTraits::SharedStorage*>(shared);
  flashinfer::SinglePrefillWithKVCacheDevice<PrefixTraits>(params, storage);
}

Status Invalid(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}

Status CudaError(const std::string& operation, cudaError_t status) {
  return Status(StatusCode::kCudaError,
                operation + ": " + cudaGetErrorString(status));
}

bool Aligned(const void* pointer, std::uint32_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

}  // namespace

struct FlashInferAttentionCommand::Impl {
  FlashInferAttentionBindings bindings;
  FlashInferAttentionVariant variant =
      FlashInferAttentionVariant::kBf16GqaDenoiseDenseBoolToBshd;
};

FlashInferAttentionCommand::FlashInferAttentionCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

FlashInferAttentionCommand::~FlashInferAttentionCommand() = default;

Status FlashInferAttentionCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    std::uint32_t active_arch, const FlashInferAttentionBindings& bindings,
    std::unique_ptr<FlashInferAttentionCommand>* output) {
  if (output == nullptr) return Invalid("FlashInfer attention command output is null");
  output->reset();
  FlashInferAttentionPayloadView parsed;
  Status status = ParseFlashInferAttentionPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (active_arch != parsed.target_arch) {
    return Status(StatusCode::kIncompatibleArchitecture,
                  "FlashInfer attention target does not match the active device");
  }
  if (bindings.query == nullptr || bindings.key == nullptr ||
      bindings.value == nullptr || bindings.mask_2d == nullptr ||
      bindings.output_bshd == nullptr) {
    return Invalid("FlashInfer attention command has a null tensor binding");
  }
  if (!Aligned(bindings.query, 16) || !Aligned(bindings.key, 16) ||
      !Aligned(bindings.value, 16) || !Aligned(bindings.output_bshd, 16)) {
    return Invalid("FlashInfer attention BF16 binding is not 16-byte aligned");
  }
  if (bindings.query_bytes < parsed.query_bytes ||
      bindings.key_bytes < parsed.key_bytes ||
      bindings.value_bytes < parsed.value_bytes ||
      bindings.mask_bytes < parsed.mask_bytes ||
      bindings.output_bytes < parsed.output_bytes) {
    return Invalid("FlashInfer attention tensor binding is shorter than the exact problem");
  }
  if (bindings.output_bshd == bindings.query ||
      bindings.output_bshd == bindings.key ||
      bindings.output_bshd == bindings.value ||
      bindings.output_bshd == bindings.mask_2d) {
    return Invalid("FlashInfer attention output must not alias an input");
  }
  const cudaError_t cuda_status =
      parsed.variant == FlashInferAttentionVariant::kBf16GqaPrefixPadBoolToBshd
          ? cudaFuncSetAttribute(AgInferFlashInferPrefixAttentionKernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(parsed.shared_bytes))
          : cudaFuncSetAttribute(AgInferFlashInferAttentionKernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(parsed.shared_bytes));
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaFuncSetAttribute(FlashInfer attention)", cuda_status);
  }
  auto impl = std::make_unique<Impl>();
  impl->bindings = bindings;
  impl->variant = parsed.variant;
  output->reset(new FlashInferAttentionCommand(std::move(impl)));
  return Status::Ok();
}

Status FlashInferAttentionCommand::Execute(void* cuda_stream) {
  if (impl_ == nullptr) return Invalid("FlashInfer attention command is not prepared");
  const bool prefix =
      impl_->variant == FlashInferAttentionVariant::kBf16GqaPrefixPadBoolToBshd;
  const std::uint32_t query_length = prefix ? 968U : 50U;
  const std::uint32_t key_length = prefix ? 968U : 1018U;
  Params params(
      reinterpret_cast<Element*>(const_cast<void*>(impl_->bindings.query)),
      reinterpret_cast<Element*>(const_cast<void*>(impl_->bindings.key)),
      reinterpret_cast<Element*>(const_cast<void*>(impl_->bindings.value)),
      reinterpret_cast<std::uint8_t*>(const_cast<void*>(impl_->bindings.mask_2d)),
      reinterpret_cast<Element*>(impl_->bindings.output_bshd),
      /*lse=*/nullptr,
      /*alibi=*/nullptr,
      /*num_qo_heads=*/8,
      /*num_kv_heads=*/1,
      /*qo_len=*/query_length,
      /*kv_len=*/key_length,
      /*q_stride_n=*/256,
      /*q_stride_h=*/query_length * 256,
      /*kv_stride_n=*/256,
      /*kv_stride_h=*/key_length * 256,
      /*head_dim=*/256,
      /*window_left=*/-1,
      /*logits_soft_cap=*/0.0F,
      /*sm_scale=*/0.0625F,
      /*rope_scale=*/1.0F,
      /*rope_theta=*/10000.0F);
  if (prefix) {
    AgInferFlashInferPrefixAttentionKernel<<<
        dim3(121, 1, 1), dim3(32, 4, 1), 49152,
        static_cast<cudaStream_t>(cuda_stream)>>>(params);
  } else {
    AgInferFlashInferAttentionKernel<<<
        dim3(7, 1, 1), dim3(32, 4, 1), 49152,
        static_cast<cudaStream_t>(cuda_stream)>>>(params);
  }
  const cudaError_t status = cudaPeekAtLastError();
  return status == cudaSuccess
             ? Status::Ok()
             : CudaError("launch FlashInfer attention", status);
}

}  // namespace aginfer::internal
