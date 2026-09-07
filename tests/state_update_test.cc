#include "state_update_payload.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef TEST_CUDA
#include "providers/state_update.h"
#include "sha256.h"
#include <cuda_runtime_api.h>
#endif
using namespace aginfer::internal;
void Check(bool value) {if(!value)throw std::runtime_error("state update contract failed");}
std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path,std::ios::binary);Check(bool(f));return {std::istreambuf_iterator<char>(f),{}};
}
void Put(std::vector<std::uint8_t>& b,int offset,std::uint64_t v) {for(int i=0;i<8;++i)b[offset+i]=(v>>(8*i))&255;}
int main(int argc,char** argv) {try {
  Check(argc>=2);auto payload=Read(argv[1]);StateUpdatePayloadView p;
  Check(ParseStateUpdatePayload(payload.data(),payload.size(),&p).ok());
  Check(p.total==1018*256&&p.count==968*256&&p.offset==0);
  auto shifted=payload;shifted.insert(shifted.begin(),0);
  Check(ParseStateUpdatePayload(shifted.data()+1,128,&p).ok());
  for(int offset:{0,8,10,12,16,24,32,80,127}) {
    auto bad=payload;bad[offset]^=1;Check(!ParseStateUpdatePayload(bad.data(),bad.size(),&p).ok());
  }
  auto bad=payload;Put(bad,32,p.total);Check(!ParseStateUpdatePayload(bad.data(),128,&p).ok());
#ifdef TEST_CUDA
  Check(argc==3);auto cubin=Read(argv[2]);
  Sha256 h;h.Update(cubin.data(),cubin.size());auto sha=h.Final();
  Put(payload,40,cubin.size());std::copy(sha.begin(),sha.end(),payload.begin()+48);
  auto made=CudaDriver::Create(120);Check(made.ok());auto driver=std::move(made.value());
  auto loaded=driver.LoadModule(cubin.data());Check(loaded.ok());
  CommandModule module{120,cubin.size(),sha,&driver,loaded.value()};
  std::array<CommandBuffer,4> buffers{};
  for(int i=0;i<4;++i) {
    buffers[i].bytes=2*p.total;Check(cudaMalloc(&buffers[i].data,buffers[i].bytes)==cudaSuccess);
    buffers[i].access=i<2?CommandOperandAccess::kRead:CommandOperandAccess::kWrite;
    Check(cudaMemset(buffers[i].data,0xa5,buffers[i].bytes)==cudaSuccess);
  }
  std::array<std::vector<std::uint16_t>,2> expected;
  for(auto& values:expected)values.assign(p.total,0xa5a5);
  cudaStream_t stream;Check(cudaStreamCreate(&stream)==cudaSuccess);
  std::unique_ptr<PreparedCommand> fill,suffix;
  Check(PrepareStateUpdate(payload,buffers,module,&fill).ok());
  auto suffix_payload=payload;Put(suffix_payload,24,50*256);Put(suffix_payload,32,968*256);
  Check(PrepareStateUpdate(suffix_payload,buffers,module,&suffix).ok());
  auto upload=[&](unsigned count,unsigned offset,unsigned seed) {
    for(unsigned side=0;side<2;++side) {
      std::vector<std::uint16_t> source(count);
      for(unsigned i=0;i<count;++i)source[i]=(i*7919+seed+side*13)&65535;
      Check(cudaMemcpy(buffers[side].data,source.data(),count*2,cudaMemcpyHostToDevice)==cudaSuccess);
      std::copy(source.begin(),source.end(),expected[side].begin()+offset);
    }
  };
  auto check=[&] {
    for(unsigned side=0;side<2;++side) {
      std::vector<std::uint16_t> actual(p.total);
      Check(cudaMemcpy(actual.data(),buffers[side+2].data,p.total*2,cudaMemcpyDeviceToHost)==cudaSuccess);
      Check(actual==expected[side]);
    }
  };
  for(unsigned observation:{1,2}) {
    upload(968*256,0,observation);Check(fill->Execute(stream).ok());Check(cudaStreamSynchronize(stream)==cudaSuccess);check();
    for(unsigned step:{1,2,3}) {
      upload(50*256,968*256,100+step);
      Check(suffix->Execute(stream).ok());Check(cudaStreamSynchronize(stream)==cudaSuccess);check();
    }
  }
  cudaGraph_t graph;cudaGraphExec_t executable;
  Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
  Check(suffix->Execute(stream).ok());Check(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
  Check(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess);
  Check(cudaGraphLaunch(executable,stream)==cudaSuccess);Check(cudaStreamSynchronize(stream)==cudaSuccess);check();
  for(int variant=0;variant<4;++variant) {
    auto b=buffers;
    if(variant==0)b[2].data=b[0].data;
    if(variant==1)b[3].data=b[2].data;
    if(variant==2)b[2].bytes=p.total*2-1;
    if(variant==3)b[2].data=static_cast<char*>(b[2].data)+2;
    std::unique_ptr<PreparedCommand> rejected;Check(!PrepareStateUpdate(payload,b,module,&rejected).ok());
  }
  // Deliberately omit the refresh: the exact judge must detect stale content.
  upload(968*256,0,77);bool detected=false;try{check();}catch(const std::runtime_error&){detected=true;}Check(detected);
  cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaStreamDestroy(stream);
  suffix.reset();fill.reset();for(auto b:buffers)cudaFree(b.data);Check(driver.UnloadModule(module.module).ok());
#endif
  std::cout<<"state update contract passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
