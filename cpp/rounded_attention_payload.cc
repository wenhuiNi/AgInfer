#include "rounded_attention_payload.h"
#include <algorithm>
#include <cstring>
namespace aginfer::internal {
Status ParseRoundedAttentionPayload(const std::uint8_t* data,std::size_t size,RoundedAttentionPayloadView* output) {
  auto bad=[] {return Status(StatusCode::kInvalidArgument,"invalid rounded-BF16 attention payload");};
  if (!data || !output || (size!=128 && size!=192)) return bad();
  *output = {};
  const bool matmul=size==192;
  const bool grouped=matmul && std::memcmp(data,"AIRAT3\0",8)==0;
  if (std::memcmp(data,grouped?"AIRAT3\0":(matmul?"AIRAT2\0":"AIRAT1\0"),8)) return bad();
  auto u32=[&](int p) {std::uint32_t x=0; for (int i=0;i<4;++i) x|=std::uint32_t(data[p+i])<<(8*i); return x;};
  if (u32(8)!=(grouped?3U:(matmul?2U:1U)) || u32(12)!=120 || u32(16)<1 || u32(16)>(matmul && !grouped?4U:2U)) return bad();
  output->softmax_warps=grouped?4U:1U;
  const auto variant=u32(16),q=variant>=3?256U:(variant==1?50U:968U),k=variant>=3?256U:(variant==1?1018U:968U);
  const auto heads=variant>=3?16U:8U,kvheads=variant>=3?16U:1U,dim=variant>=3?72U:256U;
  if (u32(20)!=q || u32(24)!=k || u32(28)!=heads || u32(32)!=kvheads || u32(36)!=dim) return bad();
  const std::uint64_t bytes=std::uint64_t(u32(40)) | (std::uint64_t(u32(44))<<32);
  if (!bytes || std::all_of(data+48,data+80,[](auto x){return x==0;})) return bad();
  if (matmul) {
    if (!u32(80) || u32(84) || u32(88)!=std::uint64_t(q)*k*heads*(variant>=3?4:2) || u32(92) ||
        std::any_of(data+168,data+192,[](auto x){return x!=0;})) return bad();
    output->cublaslt_version=u32(80); output->workspace_bytes=u32(88);
    for (int i=0;i<9;++i) {
      if (u32(96+i*4)>0x7fffffffU || u32(132+i*4)>0x7fffffffU) return bad();
      if (i>=7 && (u32(96+i*4)>65535U || u32(132+i*4)>65535U)) return bad();
      output->qk_algorithm[i]=u32(96+i*4); output->pv_algorithm[i]=u32(132+i*4);
    }
  } else if (std::any_of(data+80,data+128,[](auto x){return x!=0;})) return bad();
  output->arch=120; output->variant=variant; output->query_length=q; output->key_length=k; output->module_bytes=bytes;
  output->query_heads=heads;output->kv_heads=kvheads;output->head_dim=dim;
  std::copy(data+48,data+80,output->module_sha256.begin());
  return Status::Ok();
}
}
