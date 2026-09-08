#include "providers/rounded_attention.h"
#include "rounded_attention_payload.h"
#include <array>
#include <cublasLt.h>
#include <string>
namespace aginfer::internal {
namespace {
Status LtStatus(cublasStatus_t status) {
  return status==CUBLAS_STATUS_SUCCESS ? Status::Ok() :
      Status(StatusCode::kCudaError,"rounded attention cuBLASLt status "+std::to_string(status));
}
// All dimensions, strides and algorithm fields are fixed during Prepare.
// Interleaved C batches write BSHD directly, without an output transpose.
struct AttentionMatmul {
  cublasLtHandle_t handle=nullptr;
  cublasLtMatmulDesc_t op=nullptr;
  cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  ~AttentionMatmul() {
    if(c)cublasLtMatrixLayoutDestroy(c);
    if(b)cublasLtMatrixLayoutDestroy(b);
    if(a)cublasLtMatrixLayoutDestroy(a);
    if(op)cublasLtMatmulDescDestroy(op);
    if(handle)cublasLtDestroy(handle);
  }
  Status Prepare(int q,int k,bool qk,const std::array<std::int32_t,9>& config,bool vision,bool tf32) {
    const auto dtype=vision?CUDA_R_32F:CUDA_R_16BF;
    const auto compute=tf32?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F;
    const int dim=vision?72:256,heads=vision?16:8;
    const int ld_input=vision?1152:256;
    auto check=[](auto status){return LtStatus(status);};
    auto status=check(cublasLtCreate(&handle)); if(!status.ok())return status;
    status=check(cublasLtMatmulDescCreate(&op,compute,CUDA_R_32F));if(!status.ok())return status;
    cublasOperation_t transpose=qk?CUBLAS_OP_T:CUBLAS_OP_N;
    status=check(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));if(!status.ok())return status;
    status=check(cublasLtMatrixLayoutCreate(&a,dtype,dim,k,ld_input));if(!status.ok())return status;
    status=check(cublasLtMatrixLayoutCreate(&b,dtype,qk?dim:k,q,qk?ld_input:k));if(!status.ok())return status;
    status=check(cublasLtMatrixLayoutCreate(&c,dtype,qk?k:dim,q,qk?k:heads*dim));if(!status.ok())return status;
    const int batch=heads;
    const std::int64_t b_stride=qk?(vision?dim:q*dim):q*k;
    for(auto item:{std::pair{a,std::int64_t(vision?dim:0)},std::pair{b,b_stride},std::pair{c,qk?std::int64_t(q)*k:std::int64_t(dim)}}) {
      status=check(cublasLtMatrixLayoutSetAttribute(item.first,CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));if(!status.ok())return status;
      status=check(cublasLtMatrixLayoutSetAttribute(item.first,CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&item.second,sizeof(item.second)));if(!status.ok())return status;
    }
    status=check(cublasLtMatmulAlgoInit(handle,compute,CUDA_R_32F,dtype,dtype,dtype,dtype,config[0],&algorithm));if(!status.ok())return status;
    constexpr std::array attrs{CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,CUBLASLT_ALGO_CONFIG_STAGES_ID};
    for(std::size_t i=0;i<attrs.size();++i) {
      status=check(cublasLtMatmulAlgoConfigSetAttribute(&algorithm,attrs[i],&config[i+1],sizeof(config[i+1])));if(!status.ok())return status;
    }
    const std::uint16_t inner=config[7],cluster=config[8];
    status=check(cublasLtMatmulAlgoConfigSetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,&inner,sizeof(inner)));if(!status.ok())return status;
    status=check(cublasLtMatmulAlgoConfigSetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,&cluster,sizeof(cluster)));if(!status.ok())return status;
    cublasLtMatmulHeuristicResult_t result{};
    status=check(cublasLtMatmulAlgoCheck(handle,op,a,b,c,c,&algorithm,&result));if(!status.ok())return status;
    if(result.state!=CUBLAS_STATUS_SUCCESS || result.workspaceSize!=0)
      return Status(StatusCode::kInvalidArgument,"rounded attention fixed algorithm requires unsupported scratch");
    return Status::Ok();
  }
  Status Execute(void* left,void* right,void* output,CudaStream stream) {
    const float alpha=1,beta=0;
    return LtStatus(cublasLtMatmul(handle,op,&alpha,left,a,right,b,&beta,output,c,output,c,&algorithm,nullptr,0,reinterpret_cast<cudaStream_t>(stream)));
  }
};
class RoundedAttention final : public PreparedCommand {
 public:
  CudaDriver* driver=nullptr;
  CuFunction function=nullptr;
  std::array<void*,5> pointers{};
  std::array<std::uint32_t,3> grid{1,8,1},block{256,1,1};
  bool library_matmul=false;
  void* scratch=nullptr;
  AttentionMatmul qk,pv;
  bool NeedsLibraryPreflight() const override { return library_matmul; }
  Status Execute(CudaStream stream) override {
    if(library_matmul) {
      auto status=qk.Execute(pointers[1],pointers[0],scratch,stream);
      if(!status.ok())return status;
      void* args[]={&scratch,&pointers[3]};
      status=driver->Launch(function,grid.data(),block.data(),0,stream,args);
      if(!status.ok())return status;
      return pv.Execute(pointers[2],scratch,pointers[4],stream);
    }
    void* args[]={&pointers[0],&pointers[1],&pointers[2],&pointers[3],&pointers[4]};
    return driver->Launch(function,grid.data(),block.data(),0,stream,args);
  }
};
}
Status PrepareRoundedAttention(std::span<const std::uint8_t> payload,
    std::span<const CommandBuffer> b,const CommandBuffer& workspace,const CommandModule& module,
    std::unique_ptr<PreparedCommand>* output) {
  auto bad=[] {return Status(StatusCode::kInvalidArgument,"rounded attention buffer/module contract mismatch");};
  if (!output) return bad();
  output->reset();
  RoundedAttentionPayloadView p;
  auto status=ParseRoundedAttentionPayload(payload.data(),payload.size(),&p);
  if (!status.ok()) return status;
  if (module.arch!=p.arch || module.bytes!=p.module_bytes || module.sha256!=p.module_sha256 ||
      !module.driver || !module.module || b.size()!=5) return bad();
  if(workspace.bytes!=p.workspace_bytes || (workspace.bytes &&
     (!workspace.data || reinterpret_cast<std::uintptr_t>(workspace.data)%256)))return bad();
  if(p.cublaslt_version && cublasLtGetVersion()!=p.cublaslt_version)
    return Status(StatusCode::kIncompatibleAbi,"rounded attention cuBLASLt version mismatch");
  const std::uint64_t scalar=p.variant>=3?4:2;
  const std::uint64_t qbytes=std::uint64_t(p.query_length)*p.query_heads*p.head_dim*scalar,kbytes=std::uint64_t(p.key_length)*p.kv_heads*p.head_dim*scalar;
  const std::array<std::uint64_t,5> sizes{qbytes,kbytes,kbytes,p.variant>=3?1U:(p.variant==1?50900U:968U),qbytes};
  for (int i=0;i<5;++i) {
    if (!b[i].data || b[i].bytes<sizes[i] ||
        b[i].access!=(i==4?CommandOperandAccess::kWrite:CommandOperandAccess::kRead) ||
        (i!=3 && reinterpret_cast<std::uintptr_t>(b[i].data)%16)) return bad();
    if (i<4) {
      const auto a=reinterpret_cast<std::uintptr_t>(b[i].data),z=reinterpret_cast<std::uintptr_t>(b[4].data);
      if ((a<=z && z-a<sizes[i]) || (z<a && a-z<qbytes)) return bad();
    }
    if(workspace.bytes) {
      const auto a=reinterpret_cast<std::uintptr_t>(b[i].data),z=reinterpret_cast<std::uintptr_t>(workspace.data);
      if((a<=z && z-a<sizes[i]) || (z<a && a-z<workspace.bytes))return bad();
    }
  }
  const char* symbol=p.variant>=3?"aginfer_materialized_softmax_f32_s256":(p.cublaslt_version ? (p.variant==1?
      "aginfer_rounded_softmax_dense_s50_k1018":"aginfer_rounded_softmax_pad_s968") : (p.variant==1?
      "aginfer_rounded_attention_bf16_dense_s50_k1018":"aginfer_rounded_attention_bf16_pad_s968"));
  if(p.softmax_warps==4)symbol=p.variant==1?"aginfer_rounded_softmax_dense_s50_k1018_w4":"aginfer_rounded_softmax_pad_s968_w4";
  auto fn=module.driver->GetFunction(module.module,symbol);
  if (!fn.ok()) return fn.status();
  auto command=std::make_unique<RoundedAttention>();
  command->driver=module.driver; command->function=std::move(fn).value(); command->grid[0]=p.query_length;
  for (int i=0;i<5;++i) command->pointers[i]=b[i].data;
  if(p.cublaslt_version) {
    command->library_matmul=true;command->scratch=workspace.data;
    command->grid={(p.query_length*p.query_heads+p.softmax_warps-1)/p.softmax_warps,1,1};
    command->block={32*p.softmax_warps,1,1};
    status=command->qk.Prepare(p.query_length,p.key_length,true,p.qk_algorithm,p.variant>=3,p.variant==3);if(!status.ok())return status;
    status=command->pv.Prepare(p.query_length,p.key_length,false,p.pv_algorithm,p.variant>=3,p.variant==3);if(!status.ok())return status;
  }
  *output=std::move(command);
  return Status::Ok();
}
}
