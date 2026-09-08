// Offline staging experiment, not an attention implementation or model benchmark.
// nvcc -std=c++17 -O3 -gencode arch=compute_120,code=sm_120 tools/tma_probe.cu -lcuda -o /tmp/aginfer_tma_probe
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

void Check(cudaError_t s) {if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void Driver(CUresult s) {if(s!=CUDA_SUCCESS)throw std::runtime_error("tensor-map status "+std::to_string(s));}
void Require(bool b,const char* s) {if(!b)throw std::runtime_error(s);}
__device__ unsigned Shared(const void* p) {return static_cast<unsigned>(__cvta_generic_to_shared(p));}
template<int R,bool Swizzle>
__device__ int Index(int i) {return Swizzle ? (i ^ (((i/64)&7)*8)) : i;}

template<int R,bool Swizzle,int Mode>
__global__ void Stage(const __grid_constant__ CUtensorMap map,const std::uint16_t* input,
                      std::uint16_t* output,int rows) {
  __shared__ __align__(1024) std::uint16_t tile[2][R*64];
  __shared__ __align__(8) unsigned long long barrier[2];
  const int col=blockIdx.x*64;
  auto issue=[&](int step) {
    const int slot=step%2,base=(blockIdx.y*4+step)*R;
    if constexpr(Mode==2) {
      if(threadIdx.x==0) {
        const unsigned b=Shared(&barrier[slot]),s=Shared(tile[slot]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(b),"r"(R*128):"memory");
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            ::"r"(s),"l"(&map),"r"(col),"r"(base),"r"(b):"memory");
      }
    } else {
      for(int vector=threadIdx.x;vector<R*8;vector+=blockDim.x) {
        const int i=vector*8,row=base+i/64;
        auto* dest=reinterpret_cast<uint4*>(&tile[slot][Index<R,Swizzle>(i)]);
        const auto* source=reinterpret_cast<const uint4*>(input+(row<rows?row*256+col+i%64:0));
        if constexpr(Mode==1) {
          asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(Shared(dest)),"l"(source),"r"(row<rows?16:0):"memory");
        } else *dest=row<rows?*source:make_uint4(0,0,0,0);
      }
      if constexpr(Mode==1) asm volatile("cp.async.commit_group;":::"memory");
    }
  };
  if constexpr(Mode==2) {
    if(threadIdx.x==0)for(int i=0;i<2;++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(Shared(&barrier[i])):"memory");
    }
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
  }
  issue(0);issue(1);
  for(int step=0;step<4;++step) {
    const int slot=step%2;
    if constexpr(Mode==2) {
      unsigned ready=0;
      do {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
            :"=r"(ready):"r"(Shared(&barrier[slot])),"r"((step/2)&1):"memory");
      }while(!ready);
    } else if constexpr(Mode==1) {
      if(step<3)asm volatile("cp.async.wait_group 1;":::"memory");
      else asm volatile("cp.async.wait_group 0;":::"memory");
    }
    __syncthreads();
    const int padded=gridDim.y*4*R,base=(blockIdx.y*4+step)*R;
    for(int i=threadIdx.x;i<R*64;i+=blockDim.x)
      output[(blockIdx.z*padded+base+i/64)*256+col+i%64]=tile[slot][Index<R,Swizzle>(i)];
    __syncthreads();
    if constexpr(Mode==2)asm volatile("fence.proxy.async.shared::cta;":::"memory");
    __syncthreads();
    if(step+2<4)issue(step+2);
  }
  if constexpr(Mode==2) {
    __syncthreads();
    if(threadIdx.x==0)for(int i=0;i<2;++i)
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(Shared(&barrier[i])):"memory");
  }
}

struct Resources {
  std::uint16_t *input=nullptr,*output=nullptr;
  cudaStream_t stream=nullptr;
  std::array<cudaGraph_t,3> graph{};
  std::array<cudaGraphExec_t,3> exec{};
  cudaEvent_t begin=nullptr,end=nullptr;
  ~Resources() {
    if(stream)cudaStreamSynchronize(stream);
    for(auto p:exec)if(p)cudaGraphExecDestroy(p);
    for(auto p:graph)if(p)cudaGraphDestroy(p);
    if(input)cudaFree(input);if(output)cudaFree(output);
    if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);
    if(stream)cudaStreamDestroy(stream);
  }
};
template<int R,bool Swizzle>
void Run(int rows) {
  Resources r;Check(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));
  Check(cudaEventCreate(&r.begin));Check(cudaEventCreate(&r.end));
  const int padded=((rows+4*R-1)/(4*R))*4*R;
  const dim3 grid(4,padded/(4*R),8);
  Check(cudaMalloc(&r.input,rows*256*2));Check(cudaMalloc(&r.output,8*padded*256*2));
  CUtensorMap map{};
  cuuint64_t dims[]={256,static_cast<cuuint64_t>(rows)},strides[]={512};
  cuuint32_t box[]={64,R},element[]={1,1};
  auto encode=[&](void* ptr) {return cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,
      ptr,dims,strides,box,element,CU_TENSOR_MAP_INTERLEAVE_NONE,
      Swizzle?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);};
  Require(encode(r.input+1)!=CUDA_SUCCESS,"misaligned tensor map unexpectedly accepted");
  strides[0]=513;Require(encode(r.input)!=CUDA_SUCCESS,"misaligned stride unexpectedly accepted");
  strides[0]=512;Driver(encode(r.input));
  std::vector<std::uint16_t> input(rows*256),output(8*padded*256);
  auto fill=[&](int phase) {
    for(std::size_t i=0;i<input.size();++i)input[i]=static_cast<std::uint16_t>(i*7919+phase*173);
    Check(cudaMemcpyAsync(r.input,input.data(),input.size()*2,cudaMemcpyHostToDevice,r.stream));
    Check(cudaMemsetAsync(r.output,0xff,output.size()*2,r.stream));
  };
  auto compare=[&] {
    Check(cudaMemcpyAsync(output.data(),r.output,output.size()*2,cudaMemcpyDeviceToHost,r.stream));
    Check(cudaStreamSynchronize(r.stream));
    for(int h=0;h<8;++h)for(int row=0;row<padded;++row)for(int col=0;col<256;++col)
      Require(output[(h*padded+row)*256+col]==(row<rows?input[row*256+col]:0),"staged tile/tail/swizzle mismatch");
  };
  auto launch=[&](int mode) {
    if(mode==0)Stage<R,Swizzle,0><<<grid,128,0,r.stream>>>(map,r.input,r.output,rows);
    if(mode==1)Stage<R,Swizzle,1><<<grid,128,0,r.stream>>>(map,r.input,r.output,rows);
    if(mode==2)Stage<R,Swizzle,2><<<grid,128,0,r.stream>>>(map,r.input,r.output,rows);
    Check(cudaGetLastError());
  };
  for(int mode=0;mode<3;++mode) {
    fill(0);launch(mode);compare();
    Check(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<8;++i)launch(mode);
    Check(cudaStreamEndCapture(r.stream,&r.graph[mode]));
    Check(cudaGraphInstantiate(&r.exec[mode],r.graph[mode],0));
    for(int phase:{1,0}) {fill(phase);Check(cudaGraphLaunch(r.exec[mode],r.stream));compare();}
  }
  std::array<std::array<float,3>,3> times{};
  for(int round=0;round<3;++round)for(int step=0;step<3;++step) {
    const int mode=round%2?2-step:step;
    Check(cudaEventRecord(r.begin,r.stream));Check(cudaGraphLaunch(r.exec[mode],r.stream));
    Check(cudaEventRecord(r.end,r.stream));Check(cudaEventSynchronize(r.end));
    float ms;Check(cudaEventElapsedTime(&ms,r.begin,r.end));times[mode][round]=ms*1000/8;
  }
  std::cout<<"rows="<<rows<<" tile="<<R<<"x64 stages=2 swizzle="<<(Swizzle?128:0)<<" bitexact=true";
  for(int mode=0;mode<3;++mode) {
    std::cout<<" "<<std::array{"vector","cp_async","tma"}[mode]<<"_us=";
    for(int i=0;i<3;++i)std::cout<<(i?",":"")<<times[mode][i];
  }
  cudaFuncAttributes attr{};Check(cudaFuncGetAttributes(&attr,Stage<R,Swizzle,2>));
  std::cout<<" tma_smem="<<attr.sharedSizeBytes<<" tma_registers="<<attr.numRegs<<'\n';
}
int main() {try {
  int device;Check(cudaGetDevice(&device));cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,device));
  Require(p.major==12 && p.minor==0,"probe requires SM120");
  std::cout<<"device="<<p.name<<" smem_per_block_optin="<<p.sharedMemPerBlockOptin<<"; staging only, not attention E2E\n";
  for(int rows:{50,968,1018}) {Run<16,false>(rows);Run<16,true>(rows);Run<32,false>(rows);Run<32,true>(rows);}
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
