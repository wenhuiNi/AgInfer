#include "rounded_mul_add_payload.h"
#include <algorithm>
#include <cstring>
namespace aginfer::internal {
namespace {
std::uint64_t Read(const std::uint8_t* p, unsigned count) {
  std::uint64_t result=0;
  for(unsigned i=0;i<count;++i)result|=std::uint64_t(p[i])<<(8*i);
  return result;
}
}
Status ParseRoundedMulAddPayload(const std::uint8_t* data, std::size_t size, RoundedMulAddPayloadView* out) {
  auto bad=[] {return Status(StatusCode::kCorruptPackage,"invalid rounded multiply-add payload");};
  if(!data || !out || size!=128 || std::memcmp(data,"AIMAD1\0",8) || Read(data+8,4)!=1
      || Read(data+12,4)!=120 || !Read(data+16,8) || Read(data+16,8)>(1U<<26)
      || !Read(data+24,8) || !std::any_of(data+32,data+64,[](auto x){return x!=0;})
      || !std::all_of(data+64,data+128,[](auto x){return x==0;}))return bad();
  RoundedMulAddPayloadView p;
  p.numel=Read(data+16,8);p.module_bytes=Read(data+24,8);
  std::copy_n(data+32,32,p.module_sha256.begin());*out=p;
  return Status::Ok();
}
}
