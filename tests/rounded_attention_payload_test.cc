#include "rounded_attention_payload.h"
#include <fstream>
#include <vector>
#include <iterator>
int main(int argc,char** argv) {
  if(argc!=2) return 1;
  std::ifstream f(argv[1],std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)),{});
  if(bytes.size()!=1024) return 2;
  using namespace aginfer::internal;
  for(int variant=1;variant<=2;++variant) {
    std::vector<std::uint8_t> b(129);
    std::copy(bytes.begin()+(variant-1)*128,bytes.begin()+variant*128,b.begin()+1);
    RoundedAttentionPayloadView p;
    if(!ParseRoundedAttentionPayload(b.data()+1,128,&p).ok() || p.variant!=static_cast<unsigned>(variant) ||
       p.query_length!=(variant==1?50U:968U) || p.key_length!=(variant==1?1018U:968U) ||
       p.module_bytes!=80000 || p.module_sha256[0]!=0x88) return 3;
    for(int pos:{0,8,10,12,16,20,24,28,32,36,80,127}) {
      b[pos+1]^=1;
      if(ParseRoundedAttentionPayload(b.data()+1,128,&p).ok()) return 4;
      b[pos+1]^=1;
    }
    if(ParseRoundedAttentionPayload(b.data()+1,127,&p).ok() ||
       ParseRoundedAttentionPayload(nullptr,128,&p).ok() ||
       ParseRoundedAttentionPayload(b.data()+1,128,nullptr).ok()) return 5;
  }
  for(int variant=1;variant<=4;++variant) {
    std::vector<std::uint8_t> b(bytes.begin()+256+(variant-1)*192,bytes.begin()+256+variant*192);
    RoundedAttentionPayloadView p;
    if(!ParseRoundedAttentionPayload(b.data(),b.size(),&p).ok() ||
       p.cublaslt_version!=120803 || p.qk_algorithm[0]!=21 || p.pv_algorithm[6]!=10 ||
       p.workspace_bytes!=(variant>=3?4194304U:(variant==1?814400U:14992384U))) return 6;
    for(int pos:{0,8,10,20,84,88,92,126,162,168,191}) {
      b[pos]^=1;
      if(ParseRoundedAttentionPayload(b.data(),b.size(),&p).ok())return 7;
      b[pos]^=1;
    }
  }
  return 0;
}
