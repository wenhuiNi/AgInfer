#include "rounded_mul_add_payload.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef TEST_CUDA
#include "providers/rounded_mul_add.h"
#include "sha256.h"
#include <cuda_runtime_api.h>
#endif
using namespace aginfer::internal;
void Check(bool ok) {if(!ok)throw std::runtime_error("rounded multiply-add contract failed");}
std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path,std::ios::binary);Check(bool(f));
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void Put64(std::vector<std::uint8_t>& data,unsigned offset,std::uint64_t value) {
  for(unsigned i=0;i<8;++i)data[offset+i]=(value>>(8*i))&255;
}
int main(int argc,char** argv) {try {
  Check(argc>=2);auto payload=Read(argv[1]);RoundedMulAddPayloadView p;
  Check(ParseRoundedMulAddPayload(payload.data(),128,&p).ok()&&p.numel==51200);
  for(unsigned index:{0,8,10,12,23,64,127}) {
    auto bad=payload;bad[index]^=128;Check(!ParseRoundedMulAddPayload(bad.data(),128,&p).ok());
  }
  for(unsigned offset:{16,24,32}) {
    auto bad=payload;std::fill_n(bad.begin()+offset,offset==32?32:8,0);
    Check(!ParseRoundedMulAddPayload(bad.data(),128,&p).ok());
  }
  auto unaligned=payload;unaligned.insert(unaligned.begin(),0);
  Check(ParseRoundedMulAddPayload(unaligned.data()+1,128,&p).ok());
  Check(!ParseRoundedMulAddPayload(payload.data(),127,&p).ok());
#ifdef TEST_CUDA
  Check(argc==3);auto cubin=Read(argv[2]);auto made=CudaDriver::Create(120);Check(made.ok());
  auto driver=std::move(made.value());auto loaded=driver.LoadModule(cubin.data());Check(loaded.ok());
  Sha256 hash;hash.Update(cubin.data(),cubin.size());auto sha=hash.Final();
  Put64(payload,24,cubin.size());std::copy(sha.begin(),sha.end(),payload.begin()+32);
  CommandModule module{120,cubin.size(),sha,&driver,loaded.value()};
  auto mul=driver.GetFunction(module.module,"aginfer_mul_bf16"),add=driver.GetFunction(module.module,"aginfer_add_bf16");
  Check(mul.ok()&&add.ok());cudaStream_t stream;
  Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
  for(std::uint64_t n:{37ULL,51200ULL,131071ULL}) {
    Put64(payload,16,n);std::array<CommandBuffer,4> b{};
    std::vector<std::uint16_t> a(n),c(n),r(n),expected(n),result(n);
    for(std::size_t i=0;i<n;++i) {
      unsigned bits=i%65280;a[i]=bits+(bits>=32640?128:0);
      bits=(i*7919+173)%65280;c[i]=bits+(bits>=32640?128:0);
      bits=(i*2137+1511)%65280;r[i]=bits+(bits>=32640?128:0);
    }
    // Removing product rounding would produce 2^-14, not zero.
    a[0]=0x3f81;c[0]=0x3f81;r[0]=0xbf82;
    for(int i=0;i<4;++i) {
      b[i].bytes=n*2;b[i].access=i==3?CommandOperandAccess::kWrite:CommandOperandAccess::kRead;
      Check(cudaMalloc(&b[i].data,2*n)==cudaSuccess);
    }
    for(auto pair:{std::pair{b[0].data,a.data()},std::pair{b[1].data,c.data()},std::pair{b[2].data,r.data()}})
      Check(cudaMemcpy(pair.first,pair.second,2*n,cudaMemcpyHostToDevice)==cudaSuccess);
    Check(cudaStreamSynchronize(nullptr)==cudaSuccess);
    void *mid=nullptr,*baseline=nullptr;
    Check(cudaMalloc(&mid,2*n)==cudaSuccess);Check(cudaMalloc(&baseline,2*n)==cudaSuccess);
    std::array<std::uint32_t,3> grid{std::uint32_t(std::min<std::uint64_t>((n+255)/256,4096)),1,1},block{256,1,1};
    void* ma[]={&b[0].data,&b[1].data,&mid,&n};void* aa[]={&b[2].data,&mid,&baseline,&n};
    Check(driver.Launch(mul.value(),grid.data(),block.data(),0,stream,ma).ok());
    Check(driver.Launch(add.value(),grid.data(),block.data(),0,stream,aa).ok());
    std::unique_ptr<PreparedCommand> command;Check(PrepareRoundedMulAdd(payload,b,module,&command).ok());
    Check(command->Execute(stream).ok());Check(cudaStreamSynchronize(stream)==cudaSuccess);
    Check(cudaMemcpy(expected.data(),baseline,2*n,cudaMemcpyDeviceToHost)==cudaSuccess);Check(expected[0]==0);
    auto compare=[&]{
      Check(cudaMemcpy(result.data(),b[3].data,2*n,cudaMemcpyDeviceToHost)==cudaSuccess);
      if(result!=expected)throw std::runtime_error("rounded multiply-add output differs");
    };
    compare();cudaGraph_t graph;cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
    Check(command->Execute(stream).ok());Check(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
    Check(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess);
    for(int i=0;i<2;++i) {
      Check(cudaMemsetAsync(b[3].data,0xff,2*n,stream)==cudaSuccess);
      Check(cudaGraphLaunch(executable,stream)==cudaSuccess);Check(cudaStreamSynchronize(stream)==cudaSuccess);compare();
    }
    for(int variant=0;variant<6;++variant) {
      auto bad=b;auto bm=module;
      if(variant==0)bad[3].data=b[0].data;
      if(variant==1)bad[2].bytes-=2;
      if(variant==2)bad[1].data=static_cast<char*>(b[1].data)+1;
      if(variant==3)bad[2].access=CommandOperandAccess::kWrite;
      if(variant==4)bm.sha256[0]^=1;
      if(variant==5)bm.arch=90;
      std::unique_ptr<PreparedCommand> rejected;Check(!PrepareRoundedMulAdd(payload,bad,bm,&rejected).ok());
    }
    Check(cudaGraphExecDestroy(executable)==cudaSuccess);Check(cudaGraphDestroy(graph)==cudaSuccess);
    command.reset();for(auto item:b)Check(cudaFree(item.data)==cudaSuccess);
    Check(cudaFree(mid)==cudaSuccess);Check(cudaFree(baseline)==cudaSuccess);
    std::cout<<"bitexact numel="<<n<<" eager+Graph; product-rounding negative control passed\n";
  }
  Check(cudaStreamDestroy(stream)==cudaSuccess);Check(driver.UnloadModule(module.module).ok());
#endif
  std::cout<<"rounded multiply-add contract passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
