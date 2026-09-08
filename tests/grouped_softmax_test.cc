#include "cuda_driver.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace aginfer::internal;
void Check(bool x) {if(!x)throw std::runtime_error("grouped softmax contract failed");}
int main(int argc,char**argv) {try {
  Check(argc==2);std::ifstream f(argv[1],std::ios::binary);Check(bool(f));
  std::vector<char> cubin{std::istreambuf_iterator<char>(f),{}};
  auto made=CudaDriver::Create(120);Check(made.ok());auto driver=std::move(made.value());
  auto module=driver.LoadModule(cubin.data());Check(module.ok());
  cudaStream_t stream;Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
  for(int q:{50,968}) {
    const int k=q==50?1018:968;const std::size_t n=8*q*k,bytes=2*n;
    auto old=driver.GetFunction(module.value(),q==50?"aginfer_rounded_softmax_dense_s50_k1018":"aginfer_rounded_softmax_pad_s968");
    auto grouped=driver.GetFunction(module.value(),q==50?"aginfer_rounded_softmax_dense_s50_k1018_w4":"aginfer_rounded_softmax_pad_s968_w4");
    Check(old.ok()&&grouped.ok());
    std::vector<std::uint16_t> input(n),expected(n),result(n),first;
    std::vector<std::uint8_t> mask(q==50?q*k:q);
    for(std::size_t i=0;i<n;++i) {
      const float v=(int((i*7919)%8192)-4096)/128.f;std::uint32_t bits;std::memcpy(&bits,&v,4);input[i]=bits>>16;
    }
    std::array<void*,2> score{};void* dm=nullptr;
    constexpr int copies=8;
    for(auto& p:score)Check(cudaMalloc(&p,bytes*copies)==cudaSuccess);
    Check(cudaMalloc(&dm,mask.size())==cudaSuccess);
    auto fill=[&](int phase) {
      for(std::size_t i=0;i<mask.size();++i)mask[i]=((i+phase*3)%11)!=0;
      if(q==50)for(int i=0;i<k;++i)mask[i]=0;else mask[0]=0;
      Check(cudaMemcpyAsync(dm,mask.data(),mask.size(),cudaMemcpyHostToDevice,stream)==cudaSuccess);
      for(auto p:score)for(int copy=0;copy<copies;++copy)
        Check(cudaMemcpyAsync(static_cast<char*>(p)+copy*bytes,input.data(),bytes,cudaMemcpyHostToDevice,stream)==cudaSuccess);
    };
    auto launch=[&](int arm,int copy=0) {
      std::array<std::uint32_t,3> grid{std::uint32_t(8*q/(arm?4:1)),1,1},block{arm?128U:32U,1,1};
      void* ptr=static_cast<char*>(score[arm])+copy*bytes;
      void* args[]={&ptr,&dm};
      Check(driver.Launch(arm?grouped.value():old.value(),grid.data(),block.data(),0,stream,args).ok());
    };
    auto compare=[&](int count=1) {
      for(int copy=0;copy<count;++copy) {
      Check(cudaMemcpyAsync(expected.data(),static_cast<char*>(score[0])+copy*bytes,bytes,cudaMemcpyDeviceToHost,stream)==cudaSuccess);
      Check(cudaMemcpyAsync(result.data(),static_cast<char*>(score[1])+copy*bytes,bytes,cudaMemcpyDeviceToHost,stream)==cudaSuccess);
      Check(cudaStreamSynchronize(stream)==cudaSuccess);Check(expected==result);
      for(auto x:result)Check((x&0x7f80)!=0x7f80);
      float uniform=1.f/k;std::uint32_t bits;std::memcpy(&bits,&uniform,4);
      auto rounded=std::uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
      for(int i=0;i<k;++i)Check(result[i]==rounded); // fully masked row, finite fill semantics
      }
    };
    fill(0);launch(0);launch(1);compare();first=result;
    std::array<cudaGraph_t,2> graph{};std::array<cudaGraphExec_t,2> exec{};
    for(int arm=0;arm<2;++arm) {
      Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
      for(int copy=0;copy<copies;++copy)launch(arm,copy);
      Check(cudaStreamEndCapture(stream,&graph[arm])==cudaSuccess);Check(cudaGraphInstantiate(&exec[arm],graph[arm],0)==cudaSuccess);
    }
    for(int phase:{1,0}) {
      fill(phase);for(auto e:exec)Check(cudaGraphLaunch(e,stream)==cudaSuccess);compare(copies);
      Check((result==first)==(phase==0));
    }
    // Ignoring the mask must visibly break the fully masked row.
    std::fill(mask.begin(),mask.end(),1);
    Check(cudaMemcpyAsync(dm,mask.data(),mask.size(),cudaMemcpyHostToDevice,stream)==cudaSuccess);
    Check(cudaMemcpyAsync(score[1],input.data(),bytes,cudaMemcpyHostToDevice,stream)==cudaSuccess);
    Check(cudaGraphLaunch(exec[1],stream)==cudaSuccess);
    Check(cudaMemcpyAsync(result.data(),score[1],bytes,cudaMemcpyDeviceToHost,stream)==cudaSuccess);
    Check(cudaStreamSynchronize(stream)==cudaSuccess);Check(result!=first);
    cudaEvent_t begin,end;Check(cudaEventCreate(&begin)==cudaSuccess);Check(cudaEventCreate(&end)==cudaSuccess);
    std::cout<<"q="<<q<<" bitexact=true changed/restored=true mask-negative=true";
    for(int round=0;round<3;++round) {
      std::array<float,2> us{};
      for(int step=0;step<2;++step) {
        int arm=round%2?1-step:step;
        fill(0); // immutable original scores and identical masks; upload outside timed interval
        Check(cudaEventRecord(begin,stream)==cudaSuccess);Check(cudaGraphLaunch(exec[arm],stream)==cudaSuccess);
        Check(cudaEventRecord(end,stream)==cudaSuccess);Check(cudaEventSynchronize(end)==cudaSuccess);
        float ms;Check(cudaEventElapsedTime(&ms,begin,end)==cudaSuccess);us[arm]=1000*ms/copies;
      }
      std::cout<<" pair"<<round<<"_us="<<us[0]<<","<<us[1];
    }
    std::cout<<'\n';Check(cudaEventDestroy(begin)==cudaSuccess);Check(cudaEventDestroy(end)==cudaSuccess);
    for(auto e:exec)Check(cudaGraphExecDestroy(e)==cudaSuccess);for(auto g:graph)Check(cudaGraphDestroy(g)==cudaSuccess);
    for(auto p:score)Check(cudaFree(p)==cudaSuccess);Check(cudaFree(dm)==cudaSuccess);
  }
  Check(cudaStreamDestroy(stream)==cudaSuccess);Check(driver.UnloadModule(module.value()).ok());return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
