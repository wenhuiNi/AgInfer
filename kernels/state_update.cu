#include <cuda_runtime.h>
#include <cstdint>
extern "C" __global__ void aginfer_state_update_bf16_pair(
    const uint4* first,const uint4* second,uint4* cache_first,uint4* cache_second,
    std::uint64_t count,std::uint64_t offset) {
  const std::uint64_t index=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index<2*count) {
    const bool second_half=index>=count;
    const auto local=second_half?index-count:index;
    (second_half?cache_second:cache_first)[offset+local]=(second_half?second:first)[local];
  }
}
