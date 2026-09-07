#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {
// Correctness-first eager-BF16 arithmetic, not an online-softmax approximation.
// Q: BHSD, singleton KV: BHSD, output: BSHD. One block per query/head.
template <int QueryLength, int KeyLength, bool PadMask>
__device__ void RoundedBf16Attention(const __nv_bfloat16* q,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    const std::uint8_t* mask, __nv_bfloat16* output) {
  __shared__ float query[256];
  __shared__ float scores[1024];
  __shared__ float reduce[256];
  const int row=blockIdx.x, head=blockIdx.y, lane=threadIdx.x;
  query[lane]=__bfloat162float(q[(head*QueryLength+row)*256+lane]);
  __syncthreads();
  float maximum=-CUDART_INF_F;
  for (int token=lane; token<1024; token+=256) {
    float score=-CUDART_INF_F;
    if (token<KeyLength) {
      const bool valid=PadMask ? (mask[row] && mask[token]) : mask[row*KeyLength+token];
      score=-2.3819763e38F;
      if (valid) {
        float dot=0.0F;
        for (int d=0; d<256; ++d) dot=fmaf(query[d],__bfloat162float(k[token*256+d]),dot);
        score=__bfloat162float(__hmul_rn(__float2bfloat16_rn(dot),__float2bfloat16_rn(0.0625F)));
      }
    }
    scores[token]=score;
    maximum=fmaxf(maximum,score);
  }
  reduce[lane]=maximum;
  __syncthreads();
  for (int step=128; step; step/=2) {
    if (lane<step) reduce[lane]=fmaxf(reduce[lane],reduce[lane+step]);
    __syncthreads();
  }
  maximum=reduce[0];
  __syncthreads();
  float sum=0.0F;
  for (int token=lane; token<1024; token+=256) {
    scores[token]=expf(scores[token]-maximum);
    sum+=scores[token];
  }
  reduce[lane]=sum;
  __syncthreads();
  for (int step=128; step; step/=2) {
    if (lane<step) reduce[lane]+=reduce[lane+step];
    __syncthreads();
  }
  const float inverse=1.0F/reduce[0];
  for (int token=lane; token<1024; token+=256)
    scores[token]=__bfloat162float(__float2bfloat16_rn(scores[token]*inverse));
  __syncthreads();
  float result=0.0F;
  for (int token=0; token<KeyLength; ++token)
    result=fmaf(scores[token],__bfloat162float(v[token*256+lane]),result);
  output[(row*8+head)*256+lane]=__float2bfloat16_rn(result);
}
}  // namespace

extern "C" __global__ void aginfer_rounded_attention_bf16_pad_s968(
    const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
    const std::uint8_t* mask,__nv_bfloat16* output) {
  RoundedBf16Attention<968,968,true>(q,k,v,mask,output);
}
extern "C" __global__ void aginfer_rounded_attention_bf16_dense_s50_k1018(
    const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
    const std::uint8_t* mask,__nv_bfloat16* output) {
  RoundedBf16Attention<50,1018,false>(q,k,v,mask,output);
}

namespace {
// Explicit score/probability materialization for a library-GEMM form. The
// finite mask fill preserves mean(V) for fully masked queries. One warp
// owns a row; its reduction order matches contiguous eager softmax.
template <int Q, int K, bool Pad>
__device__ void RoundedScoreSoftmax(__nv_bfloat16* scores,
                                   const std::uint8_t* mask) {
  const int row = blockIdx.x, lane = threadIdx.x;
  float values[32], maximum = -CUDART_INF_F;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int token = lane + i * 32;
    float value = -CUDART_INF_F;
    if (token < K) {
      const bool valid = Pad ? (mask[row % Q] && mask[token])
                             : mask[(row % Q) * K + token];
      value = valid ? __bfloat162float(__hmul_rn(scores[row * K + token],
                         __float2bfloat16_rn(0.0625F))) : -2.3819763e38F;
    }
    values[i] = value;
    maximum = fmaxf(maximum, value);
  }
  for (int offset = 16; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  float sum = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    values[i] = expf(values[i] - maximum);
    sum += values[i];
  }
  for (int offset = 16; offset; offset /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int token = lane + i * 32;
    if (token < K)
      scores[row * K + token] = __float2bfloat16_rn(values[i] / sum);
  }
}
}

extern "C" __global__ void aginfer_rounded_softmax_pad_s968(
    __nv_bfloat16* scores, const std::uint8_t* mask) {
  RoundedScoreSoftmax<968,968,true>(scores, mask);
}
extern "C" __global__ void aginfer_rounded_softmax_dense_s50_k1018(
    __nv_bfloat16* scores, const std::uint8_t* mask) {
  RoundedScoreSoftmax<50,1018,false>(scores, mask);
}

extern "C" __global__ void aginfer_materialized_softmax_f32_s256(
    float* scores, const std::uint8_t* mask) {
  const int row=blockIdx.x,lane=threadIdx.x;
  float values[8],maximum=-CUDART_INF_F;
#pragma unroll
  for(int i=0;i<8;++i) {
    values[i]=mask[0] ? __fmul_rn(scores[row*256+lane+i*32],0.11785113019775793F) : -2.3819763e38F;
    maximum=fmaxf(maximum,values[i]);
  }
  for(int offset=16;offset;offset/=2)
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
  float sum=0;
#pragma unroll
  for(int i=0;i<8;++i) {values[i]=expf(values[i]-maximum);sum+=values[i];}
  for(int offset=16;offset;offset/=2)sum+=__shfl_xor_sync(0xffffffff,sum,offset);
#pragma unroll
  for(int i=0;i<8;++i)scores[row*256+lane+i*32]=values[i]/sum;
}
