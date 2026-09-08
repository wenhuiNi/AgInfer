#include "projection_split_payload.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef TEST_CUDA
#include "providers/projection_split.h"
#include "sha256.h"
#include <cuda_runtime_api.h>
#endif
using namespace aginfer::internal;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("QKV RoPE pack contract at line " + std::to_string(__LINE__)); } while (0)
std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path, std::ios::binary); CHECK(f.good());
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

int main(int argc, char** argv) { try {
  CHECK(argc >= 2);
  auto payload = Read(argv[1]);
  std::memcpy(payload.data(), "AIQRP1\0", 8);
  ProjectionSplitPayloadView p;
  CHECK(ParseProjectionSplitPayload(payload.data(), payload.size(), &p).ok());
  CHECK(p.rope_pack && p.rows == 50 && p.widths == (std::array<std::uint32_t,3>{2048,256,256}));
  auto unaligned = payload; unaligned.insert(unaligned.begin(), 0);
  CHECK(ParseProjectionSplitPayload(unaligned.data()+1,128,&p).ok());
  for (int offset : {0,8,10,12,16,20,24,28,72,127}) {
    auto bad = payload; bad[offset] ^= 8;
    CHECK(!ParseProjectionSplitPayload(bad.data(),bad.size(),&p).ok());
  }
  CHECK(!ParseProjectionSplitPayload(payload.data(),127,&p).ok());
#ifdef TEST_CUDA
  CHECK(argc == 3);
  auto cubin = Read(argv[2]);
  auto made = CudaDriver::Create(120); CHECK(made.ok()); auto driver = std::move(made.value());
  auto loaded = driver.LoadModule(cubin.data()); CHECK(loaded.ok());
  Sha256 hasher; hasher.Update(cubin.data(),cubin.size()); auto sha = hasher.Final();
  std::uint64_t size = cubin.size();
  for (unsigned i=0;i<8;++i) payload[32+i] = (size >> (8*i)) & 255;
  std::copy(sha.begin(),sha.end(),payload.begin()+40);
  CommandModule module{120,size,sha,&driver,loaded.value()};
  std::array<CommandBuffer,7> b{};
  std::array<std::size_t,7> sizes{256000,200,495616,495616,204800,521216,521216};
  for (int i=0;i<7;++i) {
    b[i].bytes=sizes[i]; b[i].access=i<4?CommandOperandAccess::kRead:CommandOperandAccess::kWrite;
    CHECK(cudaMalloc(&b[i].data,b[i].bytes)==cudaSuccess);
  }
  // Independent old four-launch chain: split, Q RoPE, K RoPE, KV pack.
  std::array<void*,7> tmp{};
  std::array<std::size_t,7> ts{204800,25600,25600,25600,204800,521216,521216};
  for(int i=0;i<7;++i) CHECK(cudaMalloc(&tmp[i],ts[i])==cudaSuccess);
  auto get = [&](const char* name) { auto fn=driver.GetFunction(module.module,name); CHECK(fn.ok()); return fn.value(); };
  auto split=get("aginfer_projection_split_bf16"), qr=get("aginfer_rope_bf16_s50_h8"),
       kr=get("aginfer_rope_bf16_s50_h1"), pack=get("aginfer_kv_pack_bf16_h1_s968_s50_d256");
  cudaStream_t stream; CHECK(cudaStreamCreate(&stream)==cudaSuccess);
  const std::array<std::uint32_t,3> block{256,1,1};
  auto launch = [&](CuFunction fn,unsigned count,void** args) {
    std::array<std::uint32_t,3> grid{count,1,1};
    CHECK(driver.Launch(fn,grid.data(),block.data(),0,stream,args).ok());
  };
  auto reference = [&] {
    std::uint32_t rows=50,qw=2048,kw=256,vw=256;
    void* sa[]={&b[0].data,&tmp[0],&tmp[1],&tmp[2],&rows,&qw,&kw,&vw}; launch(split,63,sa);
    void* qa[]={&tmp[0],&b[1].data,&tmp[4]}; launch(qr,200,qa);
    void* ka[]={&tmp[1],&b[1].data,&tmp[3]}; launch(kr,25,ka);
    void* pa[]={&b[2].data,&b[3].data,&tmp[3],&tmp[2],&tmp[5],&tmp[6]}; launch(pack,255,pa);
  };
  std::unique_ptr<PreparedCommand> command;
  CHECK(PrepareProjectionSplit(payload,b,module,&command).ok());
  std::array<std::vector<std::uint8_t>,4> inputs;
  auto fill = [&](int variant) {
    for(int slot : {0,2,3}) {
      std::vector<std::uint16_t> data(sizes[slot]/2);
      for(std::size_t i=0;i<data.size();++i) {
        // Finite Q/K, arbitrary raw prefix/V patterns including signed zero.
        auto bits=static_cast<std::uint16_t>((i*7919+slot*127+variant*73)&65535);
        data[i]=slot==0 ? (bits&0x807f)|0x3f00 : bits;
      }
      inputs[slot].resize(sizes[slot]); std::memcpy(inputs[slot].data(),data.data(),sizes[slot]);
    }
    std::array<std::int32_t,50> positions{};
    for(int i=0;i<50;++i) positions[i]=(i%3==0?-1:i%3==1?0:900+i*17)+variant*137;
    inputs[1].resize(200); std::memcpy(inputs[1].data(),positions.data(),200);
    for(int i=0;i<4;++i) CHECK(cudaMemcpy(b[i].data,inputs[i].data(),sizes[i],cudaMemcpyHostToDevice)==cudaSuccess);
    for(int i=4;i<7;++i) CHECK(cudaMemset(b[i].data,0xff,sizes[i])==cudaSuccess);
  };
  auto compare = [&] {
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess);
    std::vector<std::uint8_t> combined;
    for(int i=4;i<7;++i) {
      std::vector<std::uint8_t> actual(sizes[i]),expected(sizes[i]);
      CHECK(cudaMemcpy(actual.data(),b[i].data,sizes[i],cudaMemcpyDeviceToHost)==cudaSuccess);
      CHECK(cudaMemcpy(expected.data(),tmp[i],sizes[i],cudaMemcpyDeviceToHost)==cudaSuccess);
      CHECK(actual==expected); combined.insert(combined.end(),actual.begin(),actual.end());
    }
    for(int i=0;i<4;++i) {
      std::vector<std::uint8_t> after(sizes[i]);
      CHECK(cudaMemcpy(after.data(),b[i].data,sizes[i],cudaMemcpyDeviceToHost)==cudaSuccess);
      CHECK(after==inputs[i]);
    }
    return combined;
  };
  fill(0); reference(); CHECK(command->Execute(stream).ok()); auto original=compare();
  cudaGraph_t graph; cudaGraphExec_t executable;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
  CHECK(command->Execute(stream).ok());
  CHECK(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
  std::size_t nodes=0; CHECK(cudaGraphGetNodes(graph,nullptr,&nodes)==cudaSuccess); CHECK(nodes==1);
  CHECK(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess);
  for(int variant : {0,1,0}) {
    fill(variant); reference();
    CHECK(cudaGraphLaunch(executable,stream)==cudaSuccess);
    auto actual=compare(); CHECK((actual==original)==(variant==0));
  }
  {
    // The old KV pack permits read-only aliases; the fused binder must too.
    auto same_prefix=b; same_prefix[3].data=b[2].data;
    std::unique_ptr<PreparedCommand> aliased;
    CHECK(PrepareProjectionSplit(payload,same_prefix,module,&aliased).ok());
    inputs[3]=inputs[2];
    CHECK(cudaMemcpy(b[3].data,inputs[3].data(),sizes[3],cudaMemcpyHostToDevice)==cudaSuccess);
    reference(); CHECK(aliased->Execute(stream).ok()); compare();
  }
  // Every short/access/misaligned/null operand is rejected; all output aliases
  // (including aligned partial overlaps) and module/arch mismatches are refused.
  for(int i=0;i<7;++i) for(int kind=0;kind<4;++kind) {
    auto bad=b;
    if(kind==0) --bad[i].bytes;
    if(kind==1) bad[i].access=CommandOperandAccess::kReadWrite;
    if(kind==2) bad[i].data=static_cast<char*>(bad[i].data)+2;
    if(kind==3) bad[i].data=nullptr;
    std::unique_ptr<PreparedCommand> rejected; CHECK(!PrepareProjectionSplit(payload,bad,module,&rejected).ok());
  }
  for(int i=4;i<7;++i) for(int j=0;j<i;++j) for(int offset : {0,16}) {
    auto bad=b; bad[i].data=static_cast<char*>(b[j].data)+offset;
    std::unique_ptr<PreparedCommand> rejected; CHECK(!PrepareProjectionSplit(payload,bad,module,&rejected).ok());
  }
  for(int kind=0;kind<3;++kind) {
    auto bad=module;
    if(kind==0) bad.arch=90;
    if(kind==1) ++bad.bytes;
    if(kind==2) bad.sha256[0]^=1;
    std::unique_ptr<PreparedCommand> rejected; CHECK(!PrepareProjectionSplit(payload,b,bad,&rejected).ok());
  }
  cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
  command.reset(); for(auto item:b) cudaFree(item.data); for(auto ptr:tmp) cudaFree(ptr);
  CHECK(driver.UnloadModule(module.module).ok());
#endif
  std::cout << "QKV RoPE pack contract passed\n"; return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
