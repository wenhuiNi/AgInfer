#include "gelu_mul_payload.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef TEST_CUDA
#include "providers/gelu_mul.h"
#include "sha256.h"
#include <cuda_runtime_api.h>
#endif
using namespace aginfer::internal;
void Check(bool ok) { if (!ok) throw std::runtime_error("GELU-multiply contract failed"); }
std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path, std::ios::binary); Check(bool(f));
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void Put64(std::vector<std::uint8_t>& data, unsigned offset, std::uint64_t value) {
  for (unsigned i=0;i<8;++i) data[offset+i]=(value>>(8*i))&255;
}
int main(int argc, char** argv) { try {
  Check(argc >= 2);
  auto payload = Read(argv[1]);
  GeluMulPayloadView p;
  Check(ParseGeluMulPayload(payload.data(),payload.size(),&p).ok());
  Check(p.numel==204800);
  auto unaligned=payload; unaligned.insert(unaligned.begin(),0);
  Check(ParseGeluMulPayload(unaligned.data()+1,128,&p).ok());
  for(unsigned offset:{0,8,10,12,23,64,127}) {
    auto bad=payload; bad[offset]^=128;
    Check(!ParseGeluMulPayload(bad.data(),bad.size(),&p).ok());
  }
  for(unsigned offset:{16,24,32}) {
    auto bad=payload; std::fill_n(bad.begin()+offset,offset==32?32:8,0);
    Check(!ParseGeluMulPayload(bad.data(),bad.size(),&p).ok());
  }
  Check(!ParseGeluMulPayload(payload.data(),127,&p).ok());
  auto packed_contract=payload; packed_contract[4]='P'; Put64(packed_contract,64,4096);
  Check(ParseGeluMulPayload(packed_contract.data(),128,&p).ok() && p.packed_width==4096);
  auto tiled_contract=packed_contract;tiled_contract[4]='T';
  Check(ParseGeluMulPayload(tiled_contract.data(),128,&p).ok() && p.packed_tiled);
  for(auto rows:{129ULL,968ULL,2048ULL}) {
    auto large=tiled_contract; Put64(large,16,rows*16384); Put64(large,64,16384);
    Check(ParseGeluMulPayload(large.data(),128,&p).ok() && p.numel==rows*16384);
  }
  auto too_many=tiled_contract; Put64(too_many,16,2049*4096ULL);
  Check(!ParseGeluMulPayload(too_many.data(),128,&p).ok());
  for(auto width:{0ULL,1ULL,32769ULL}) {
    auto bad=packed_contract; Put64(bad,64,width);
    Check(!ParseGeluMulPayload(bad.data(),128,&p).ok());
  }
#ifdef TEST_CUDA
  Check(argc==3);
  auto cubin=Read(argv[2]);
  auto made=CudaDriver::Create(120); Check(made.ok()); auto driver=std::move(made.value());
  auto loaded=driver.LoadModule(cubin.data()); Check(loaded.ok());
  Sha256 hasher; hasher.Update(cubin.data(),cubin.size()); auto sha=hasher.Final();
  Put64(payload,24,cubin.size()); std::copy(sha.begin(),sha.end(),payload.begin()+32);
  CommandModule module{120,cubin.size(),sha,&driver,loaded.value()};
  auto gelu=driver.GetFunction(module.module,"aginfer_gelu_tanh_bf16"); Check(gelu.ok());
  auto mul=driver.GetFunction(module.module,"aginfer_mul_bf16"); Check(mul.ok());
  cudaStream_t stream; Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
  for(std::uint64_t n:{std::uint64_t(37),std::uint64_t(204800),std::uint64_t(15859712)}) {
    Put64(payload,16,n);
    std::array<CommandBuffer,3> b{};
    std::vector<std::uint16_t> input(n),up(n),reference(n),result(n);
    // Cover every finite BF16 bit pattern, including signed zero/subnormals,
    // plus a decorrelated second operand. Synthetic local kernel checks only.
    for(std::size_t i=0;i<n;++i) {
      unsigned bits=i%65280; input[i]=bits+(bits>=32640?128:0);
      bits=(i*7919+173)%65280; up[i]=bits+(bits>=32640?128:0);
    }
    for(int i=0;i<3;++i) {
      b[i].bytes=n*2; b[i].access=i==2?CommandOperandAccess::kWrite:CommandOperandAccess::kRead;
      Check(cudaMalloc(&b[i].data,b[i].bytes)==cudaSuccess);
    }
    void *intermediate=nullptr,*baseline=nullptr;
    Check(cudaMalloc(&intermediate,n*2)==cudaSuccess); Check(cudaMalloc(&baseline,n*2)==cudaSuccess);
    Check(cudaMemcpy(b[0].data,input.data(),n*2,cudaMemcpyHostToDevice)==cudaSuccess);
    Check(cudaMemcpy(b[1].data,up.data(),n*2,cudaMemcpyHostToDevice)==cudaSuccess);
    Check(cudaStreamSynchronize(nullptr)==cudaSuccess);
    std::array<std::uint32_t,3> grid{std::uint32_t(std::min<std::uint64_t>((n+255)/256,4096)),1,1},block{256,1,1};
    void* ga[]={&b[0].data,&intermediate,&n};
    void* ma[]={&intermediate,&b[1].data,&baseline,&n};
    Check(driver.Launch(gelu.value(),grid.data(),block.data(),0,stream,ga).ok());
    Check(driver.Launch(mul.value(),grid.data(),block.data(),0,stream,ma).ok());
    std::unique_ptr<PreparedCommand> command;
    Check(PrepareGeluMul(payload,b,module,&command).ok());
    Check(command->Execute(stream).ok()); Check(cudaStreamSynchronize(stream)==cudaSuccess);
    Check(cudaMemcpy(reference.data(),baseline,n*2,cudaMemcpyDeviceToHost)==cudaSuccess);
    auto compare=[&] {
      Check(cudaMemcpy(result.data(),b[2].data,n*2,cudaMemcpyDeviceToHost)==cudaSuccess);
      if(result!=reference) {
        auto mismatch=std::mismatch(result.begin(),result.end(),reference.begin());
        throw std::runtime_error("GELU-multiply differs at element "+std::to_string(mismatch.first-result.begin()));
      }
    };
    compare();
    for(char form:{'P','T'}) {
      auto packed_payload=payload;
      packed_payload[4]=form; Put64(packed_payload,64,n==37?37:n==204800?4096:16384);
      GeluMulPayloadView pp;
      Check(ParseGeluMulPayload(packed_payload.data(),128,&pp).ok());
      std::vector<std::uint16_t> packed(2*n);
      for(std::size_t i=0;i<n;++i) {
        auto j=(i/pp.packed_width)*(2*pp.packed_width)+i%pp.packed_width;
        packed[j]=input[i]; packed[j+pp.packed_width]=up[i];
      }
      std::array<CommandBuffer,2> pb{CommandBuffer{nullptr,4*n,CommandOperandAccess::kRead},b[2]};
      Check(cudaMalloc(&pb[0].data,4*n)==cudaSuccess);
      Check(cudaMemcpy(pb[0].data,packed.data(),4*n,cudaMemcpyHostToDevice)==cudaSuccess);
      Check(cudaStreamSynchronize(nullptr)==cudaSuccess);
      std::unique_ptr<PreparedCommand> pc;
      Check(PrepareGeluMul(packed_payload,pb,module,&pc).ok());
      Check(pc->Execute(stream).ok()); Check(cudaStreamSynchronize(stream)==cudaSuccess); compare();
      cudaGraph_t pg; cudaGraphExec_t pe;
      Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
      Check(pc->Execute(stream).ok()); Check(cudaStreamEndCapture(stream,&pg)==cudaSuccess);
      Check(cudaGraphInstantiate(&pe,pg,0)==cudaSuccess);
      Check(cudaMemsetAsync(pb[1].data,0xff,2*n,stream)==cudaSuccess);
      Check(cudaGraphLaunch(pe,stream)==cudaSuccess); Check(cudaStreamSynchronize(stream)==cudaSuccess); compare();
      for(int variant=0;variant<3;++variant) {
        auto bad=pb;
        if(variant==0)bad[0].bytes-=2;
        if(variant==1)bad[1].data=static_cast<char*>(bad[0].data)+2*n;
        if(variant==2)bad[0].access=CommandOperandAccess::kWrite;
        std::unique_ptr<PreparedCommand> rejected;
        Check(!PrepareGeluMul(packed_payload,bad,module,&rejected).ok());
      }
      Check(cudaGraphExecDestroy(pe)==cudaSuccess); Check(cudaGraphDestroy(pg)==cudaSuccess);
      pc.reset(); Check(cudaFree(pb[0].data)==cudaSuccess);
      std::cout<<"packed form="<<form<<" bitexact numel="<<n<<'\n';
    }
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
    Check(command->Execute(stream).ok());
    Check(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
    Check(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess);
    for(int repeat=0;repeat<2;++repeat) {
      Check(cudaMemsetAsync(b[2].data,0xff,n*2,stream)==cudaSuccess);
      Check(cudaGraphLaunch(executable,stream)==cudaSuccess); Check(cudaStreamSynchronize(stream)==cudaSuccess); compare();
    }
    for(int variant=0;variant<6;++variant) {
      auto bad=b; auto bad_module=module;
      if(variant==0)bad[2].data=b[0].data;
      if(variant==1)bad[1].bytes-=2;
      if(variant==2)bad[1].data=static_cast<char*>(bad[1].data)+1;
      if(variant==3)bad[0].access=CommandOperandAccess::kWrite;
      if(variant==4)bad_module.sha256[0]^=1;
      if(variant==5)bad_module.arch=90;
      std::unique_ptr<PreparedCommand> rejected;
      Check(!PrepareGeluMul(payload,bad,bad_module,&rejected).ok()&&!rejected);
    }
    Check(cudaGraphExecDestroy(executable)==cudaSuccess); Check(cudaGraphDestroy(graph)==cudaSuccess);
    command.reset(); for(auto item:b)Check(cudaFree(item.data)==cudaSuccess);
    Check(cudaFree(intermediate)==cudaSuccess); Check(cudaFree(baseline)==cudaSuccess);
    std::cout<<"bitexact numel="<<n<<" eager+graph repeats passed\n";
  }
  Check(cudaStreamDestroy(stream)==cudaSuccess); Check(driver.UnloadModule(module.module).ok());
#endif
  std::cout<<"GELU-multiply contract passed\n"; return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; } }
