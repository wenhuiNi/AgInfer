#include "state_update_payload.h"
#include <algorithm>
#include <cstring>
namespace aginfer::internal {
Status ParseStateUpdatePayload(const std::uint8_t* data,std::size_t size,StateUpdatePayloadView* out) {
  auto bad=[] {return Status(StatusCode::kCorruptPackage,"invalid state update payload");};
  if(!data||!out||size!=128||std::memcmp(data,"AISTU1\0",8))return bad();
  auto read=[&](int offset,int n) {std::uint64_t v=0;for(int i=0;i<n;++i)v|=std::uint64_t(data[offset+i])<<(8*i);return v;};
  if(read(8,4)!=1||read(12,4)!=120||!std::all_of(data+80,data+128,[](auto x){return x==0;}))return bad();
  StateUpdatePayloadView p{read(16,8),read(24,8),read(32,8),read(40,8),{}};
  if(!p.count||p.count>p.total||p.total>(1ULL<<30)||p.offset>p.total-p.count
      ||p.total%8||p.count%8||p.offset%8||!p.module_bytes)return bad();
  std::copy_n(data+48,32,p.module_sha256.begin());
  if(!std::any_of(p.module_sha256.begin(),p.module_sha256.end(),[](auto x){return x!=0;}))return bad();
  *out=p;return Status::Ok();
}
}
