#include "projection_split_payload.h"
#include <algorithm>
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
void Check(bool ok) { if (!ok) throw std::runtime_error("projection split contract failed"); }
std::vector<std::uint8_t> Read(const char* path) {
  std::ifstream f(path, std::ios::binary); Check(bool(f));
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
int main(int argc, char** argv) { try {
  Check(argc >= 2);
  auto payload = Read(argv[1]);
  ProjectionSplitPayloadView p;
  Check(ParseProjectionSplitPayload(payload.data(), payload.size(), &p).ok());
  Check(p.rows == 50 && p.widths == std::array<std::uint32_t,3>{2048,256,256});
  auto unaligned = payload; unaligned.insert(unaligned.begin(), 0);
  Check(ParseProjectionSplitPayload(unaligned.data()+1, 128, &p).ok());
  for (unsigned offset : {0,8,10,12,16,20,72,127}) {
    auto bad = payload; bad[offset] ^= offset == 20 ? 1 : 128;
    Check(!ParseProjectionSplitPayload(bad.data(), bad.size(), &p).ok());
  }
  Check(!ParseProjectionSplitPayload(payload.data(), 127, &p).ok());
#ifdef TEST_CUDA
  Check(argc == 3);
  auto cubin = Read(argv[2]);
  auto made = CudaDriver::Create(120); Check(made.ok()); auto driver = std::move(made.value());
  auto loaded = driver.LoadModule(cubin.data()); Check(loaded.ok());
  Sha256 hasher; hasher.Update(cubin.data(), cubin.size()); auto sha = hasher.Final();
  std::uint64_t size = cubin.size();
  for (unsigned i=0;i<8;++i) payload[32+i] = (size >> (8*i)) & 255;
  std::copy(sha.begin(), sha.end(), payload.begin()+40);
  CommandModule module{120,size,sha,&driver,loaded.value()};
  std::array<CommandBuffer,4> b{};
  std::vector<std::uint16_t> input(50*2560);
  for (std::size_t i=0;i<input.size();++i) input[i] = (i*7919) & 65535;
  for (int i=0;i<4;++i) {
    b[i].bytes = (i==0 ? input.size() : std::size_t(50)*p.widths[i-1])*2;
    Check(cudaMalloc(&b[i].data,b[i].bytes)==cudaSuccess);
    b[i].access = i==0 ? CommandOperandAccess::kRead : CommandOperandAccess::kWrite;
  }
  Check(cudaMemcpy(b[0].data,input.data(),b[0].bytes,cudaMemcpyHostToDevice)==cudaSuccess);
  std::unique_ptr<PreparedCommand> command;
  Check(PrepareProjectionSplit(payload,b,module,&command).ok());
  cudaStream_t stream; Check(cudaStreamCreate(&stream)==cudaSuccess);
  Check(command->Execute(stream).ok()); Check(cudaStreamSynchronize(stream)==cudaSuccess);
  auto check = [&] {
    unsigned start=0;
    for(int j=1;j<4;++j) {
      std::vector<std::uint16_t> output(b[j].bytes/2);
      Check(cudaMemcpy(output.data(),b[j].data,b[j].bytes,cudaMemcpyDeviceToHost)==cudaSuccess);
      for(unsigned row=0;row<50;++row) for(unsigned col=0;col<p.widths[j-1];++col)
        Check(output[row*p.widths[j-1]+col]==input[row*2560+start+col]);
      start+=p.widths[j-1];
    }
  };
  check();
  cudaGraph_t graph; cudaGraphExec_t executable;
  Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
  Check(command->Execute(stream).ok());
  Check(cudaStreamEndCapture(stream,&graph)==cudaSuccess);
  Check(cudaGraphInstantiate(&executable,graph,0)==cudaSuccess);
  Check(cudaGraphLaunch(executable,stream)==cudaSuccess); Check(cudaStreamSynchronize(stream)==cudaSuccess); check();
  for(int variant=0;variant<3;++variant) {
    auto bad=b;
    if(variant==0) bad[1].data=b[0].data;
    if(variant==1) bad[1].bytes-=2;
    if(variant==2) bad[1].data=static_cast<char*>(bad[1].data)+2;
    std::unique_ptr<PreparedCommand> rejected;
    Check(!PrepareProjectionSplit(payload,bad,module,&rejected).ok());
  }
  cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
  command.reset(); for(auto item:b) cudaFree(item.data); Check(driver.UnloadModule(module.module).ok());
#endif
  std::cout << "projection split contract passed\n";
  return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }
