#include "compact_gate_payload.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef TEST_CUDA
#include "providers/compact_gate.h"
#include "providers/rounded_mul_add.h"
#include "sha256.h"
#include <cuda_runtime_api.h>
#endif
using namespace aginfer::internal;
void Check(bool ok) {if(!ok)throw std::runtime_error("compact-gate contract failed");}
std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path,std::ios::binary);Check(bool(f));
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
int main(int argc,char** argv) {try {
  Check(argc>=2);auto base=Read(argv[1]);CompactGatePayloadView p;
  Check(!ParseCompactGatePayload(base.data(),base.size(),&p).ok());
  for(auto magic:{"AIANG1\0","AIGRD1\0","AIGRN1\0"}) {
    auto data=base;std::memcpy(data.data(),magic,8);
    Check(ParseCompactGatePayload(data.data(),data.size(),&p).ok());
    for(int i:{0,8,10,12,16,64,127}) {
      auto bad=data;bad[i]^=1;Check(!ParseCompactGatePayload(bad.data(),bad.size(),&p).ok());
    }
    Check(!ParseCompactGatePayload(data.data(),127,&p).ok());
  }
#ifdef TEST_CUDA
  Check(argc==3);auto cubin=Read(argv[2]);auto made=CudaDriver::Create(120);Check(made.ok());
  auto driver=std::move(made.value());auto loaded=driver.LoadModule(cubin.data());Check(loaded.ok());
  Sha256 hash;hash.Update(cubin.data(),cubin.size());auto sha=hash.Final();
  auto size=std::uint64_t(cubin.size());for(int j=0;j<8;++j)base[24+j]=(size>>(8*j))&255;
  std::copy(sha.begin(),sha.end(),base.begin()+32);
  auto norm=base,res=base,fused=base;std::memcpy(norm.data(),"AIANG1\0",8);std::memcpy(res.data(),"AIGRD1\0",8);std::memcpy(fused.data(),"AIGRN1\0",8);
  CommandModule module{120,cubin.size(),sha,&driver,loaded.value()};
  // hidden, modulation, new norm, old norm, old gate, activation, residual, new out, old out
  std::array<CommandBuffer,13> b{};
  for(int i=0;i<13;++i) {
    b[i].bytes=(i==1||i==9)?12288:102400;b[i].access=CommandOperandAccess::kRead;
    Check(cudaMalloc(&b[i].data,b[i].bytes)==cudaSuccess);
  }
  auto write=[](CommandBuffer x){x.access=CommandOperandAccess::kWrite;return x;};
  std::array nb{b[0],b[1],write(b[2])};
  std::array rb{b[5],b[1],b[6],write(b[7])},ob{b[5],b[4],b[6],write(b[8])};
  std::unique_ptr<PreparedCommand> nc,rc,oc;
  Check(PrepareCompactGate(norm,nb,module,&nc).ok());Check(PrepareCompactGate(res,rb,module,&rc).ok());
  Check(PrepareRoundedMulAdd(base,ob,module,&oc).ok());
  std::array fb{b[5],b[1],b[6],b[9],write(b[10]),write(b[11])};
  std::array snb{b[8],b[9],write(b[12])};
  std::unique_ptr<PreparedCommand> fc,sc;
  Check(PrepareCompactGate(fused,fb,module,&fc).ok());Check(PrepareCompactGate(norm,snb,module,&sc).ok());
  auto old=driver.GetFunction(module.module,"aginfer_adaptive_rms_norm_bf16_f32_1024");Check(old.ok());
  cudaStream_t stream;Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
  std::array<std::uint32_t,3> grid{50,1,1},block{256,1,1};
  void* args[]={&b[0].data,&b[1].data,&b[3].data,&b[4].data};
  std::vector<std::uint16_t> x(51200),r(51200),out(51200),expected(51200),first;
  std::vector<float> mod(3072);
  auto fill=[&](int phase) {
    for(int i=0;i<51200;++i) {x[i]=0x3e00+(i*13%255);r[i]=0xbe00+(i*19%255);}
    for(int i=0;i<3072;++i)mod[i]=((i*37+phase*31)%1009-504)/1024.f;
    // Gate cast and product rounding are independently observable in this fixture.
    x[0]=0x3f81;r[0]=0xbf82;mod[2048]=1.0078125f;
    for(int i:{0,5})Check(cudaMemcpyAsync(b[i].data,x.data(),102400,cudaMemcpyHostToDevice,stream)==cudaSuccess);
    Check(cudaMemcpyAsync(b[6].data,r.data(),102400,cudaMemcpyHostToDevice,stream)==cudaSuccess);
    Check(cudaMemcpyAsync(b[1].data,mod.data(),12288,cudaMemcpyHostToDevice,stream)==cudaSuccess);
    auto next=mod;for(auto& v:next)v=v*.75f+.125f;
    Check(cudaMemcpyAsync(b[9].data,next.data(),12288,cudaMemcpyHostToDevice,stream)==cudaSuccess);
    Check(cudaStreamSynchronize(stream)==cudaSuccess);
  };
  auto compare=[&] {
    for(auto pair:{std::pair{10,7},std::pair{11,12},std::pair{2,3},std::pair{7,8}}) {
      Check(cudaMemcpy(out.data(),b[pair.first].data,102400,cudaMemcpyDeviceToHost)==cudaSuccess);
      Check(cudaMemcpy(expected.data(),b[pair.second].data,102400,cudaMemcpyDeviceToHost)==cudaSuccess);
      Check(out==expected);
    }
    Check(out[0]==0);
  };
  fill(0);
  Check(driver.Launch(old.value(),grid.data(),block.data(),0,stream,args).ok());Check(oc->Execute(stream).ok());
  Check(nc->Execute(stream).ok());Check(rc->Execute(stream).ok());Check(sc->Execute(stream).ok());Check(fc->Execute(stream).ok());Check(cudaStreamSynchronize(stream)==cudaSuccess);
  compare();first=out;
  cudaGraph_t graph;cudaGraphExec_t exec;
  Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
  Check(nc->Execute(stream).ok());Check(rc->Execute(stream).ok());Check(fc->Execute(stream).ok());
  Check(cudaStreamEndCapture(stream,&graph)==cudaSuccess);Check(cudaGraphInstantiate(&exec,graph,0)==cudaSuccess);
  for(int phase:{1,0}) {
    fill(phase);
    Check(driver.Launch(old.value(),grid.data(),block.data(),0,stream,args).ok());Check(oc->Execute(stream).ok());
    Check(sc->Execute(stream).ok());
    for(int i:{2,7,10,11})Check(cudaMemsetAsync(b[i].data,0xff,102400,stream)==cudaSuccess);
    Check(cudaGraphLaunch(exec,stream)==cudaSuccess);Check(cudaStreamSynchronize(stream)==cudaSuccess);compare();
    Check((out==first)==(phase==0));
  }
  // Detect removal of either rounding boundary with scalar negative controls.
  auto bf=[](float f){std::uint32_t u;std::memcpy(&u,&f,4);return std::uint16_t((u+0x7fff+((u>>16)&1))>>16);};
  auto fp=[](std::uint16_t x){auto u=std::uint32_t(x)<<16;float f;std::memcpy(&f,&u,4);return f;};
  int no_gate=0,no_product=0;
  for(int i=0;i<51200;++i) {
    const float a=fp(x[i]),g=mod[2048+i%1024],v=fp(r[i]);
    const auto correct=bf(fp(bf(a*fp(bf(g))))+v);
    no_gate+=bf(fp(bf(a*g))+v)!=correct;
    no_product+=bf(a*fp(bf(g))+v)!=correct;
  }
  Check(no_gate>0 && no_product>0);
  std::cout<<"eager/changed/restored Graph bitexact; negative gate="<<no_gate<<" product="<<no_product<<'\n';
  for(int kind:{0,1,2}) {
    auto payload=kind==2?fused:kind==0?norm:res;
    std::vector<CommandBuffer> bindings=kind==2?std::vector<CommandBuffer>(fb.begin(),fb.end()):kind==0?std::vector<CommandBuffer>(nb.begin(),nb.end()):std::vector<CommandBuffer>(rb.begin(),rb.end());
    for(std::size_t i=0;i<bindings.size();++i) {
      for(int mode=0;mode<3;++mode) {
        auto bad=bindings;
        if(mode==0)bad[i].bytes=1;
        if(mode==1)bad[i].data=static_cast<char*>(bad[i].data)+1;
        if(mode==2)bad[i].access=CommandOperandAccess::kReadWrite;
        std::unique_ptr<PreparedCommand> invalid;Check(!PrepareCompactGate(payload,bad,module,&invalid).ok());
      }
    }
    for(std::size_t i=0;i+1<bindings.size();++i) {
      auto bad=bindings;bad.back().data=bad[i].data;
      std::unique_ptr<PreparedCommand> invalid;Check(!PrepareCompactGate(payload,bad,module,&invalid).ok());
    }
    if(kind==2)for(int i:{0,1,2,3,5}){auto bad=bindings;bad[4].data=bad[i].data;
      std::unique_ptr<PreparedCommand> invalid;Check(!PrepareCompactGate(payload,bad,module,&invalid).ok());}
    auto wrong=module;wrong.arch=90;std::unique_ptr<PreparedCommand> invalid;
    Check(!PrepareCompactGate(payload,bindings,wrong,&invalid).ok());
    wrong=module;wrong.sha256[0]^=1;Check(!PrepareCompactGate(payload,bindings,wrong,&invalid).ok());
  }
  Check(cudaGraphExecDestroy(exec)==cudaSuccess);Check(cudaGraphDestroy(graph)==cudaSuccess);
  nc.reset();rc.reset();oc.reset();fc.reset();sc.reset();for(auto v:b)Check(cudaFree(v.data)==cudaSuccess);
  Check(cudaStreamDestroy(stream)==cudaSuccess);Check(driver.UnloadModule(module.module).ok());
#endif
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
