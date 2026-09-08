#include "providers/compact_gate.h"
#include "compact_gate_payload.h"
namespace aginfer::internal {
namespace {
class CompactGate final : public PreparedCommand {
 public:
  CudaDriver* driver=nullptr; CuFunction function=nullptr;
  std::array<void*,4> pointers{};
  std::array<std::uint32_t,3> grid{},block{256,1,1};
  Status Execute(CudaStream stream) override {
    void* args[]={&pointers[0],&pointers[1],&pointers[2],&pointers[3]};
    return driver->Launch(function,grid.data(),block.data(),0,stream,args);
  }
};
}
Status PrepareCompactGate(std::span<const std::uint8_t> payload,std::span<const CommandBuffer> b,
                          const CommandModule& module,std::unique_ptr<PreparedCommand>* out) {
  auto bad=[] {return Status(StatusCode::kInvalidArgument,"compact-gate binding mismatch");};
  if(!out)return bad();
  out->reset();CompactGatePayloadView p;
  auto status=ParseCompactGatePayload(payload.data(),payload.size(),&p);
  if(!status.ok())return status;
  const int count=p.norm?3:4;
  if(module.arch!=120 || module.bytes!=p.module_bytes || module.sha256!=p.module_sha256
      || !module.driver || !module.module || b.size()!=static_cast<std::size_t>(count))return bad();
  for(int i=0;i<count;++i) {
    const auto bytes=i==1?12288U:102400U;
    const auto align=p.norm?16U:(i==1?4U:2U);
    if(!b[i].data || reinterpret_cast<std::uintptr_t>(b[i].data)%align || b[i].bytes<bytes
        || b[i].access!=(i==count-1?CommandOperandAccess::kWrite:CommandOperandAccess::kRead))return bad();
  }
  const auto output=reinterpret_cast<std::uintptr_t>(b[count-1].data);
  for(int i=0;i<count-1;++i) {
    const auto input=reinterpret_cast<std::uintptr_t>(b[i].data);
    if(output<=input?input-output<102400:output-input<(i==1?12288U:102400U))return bad();
  }
  auto f=module.driver->GetFunction(module.module,p.norm?"aginfer_adaptive_rms_norm_no_gate_bf16_f32_1024"
                                                        :"aginfer_modulated_residual_bf16_f32_1024");
  if(!f.ok())return f.status();
  auto c=std::make_unique<CompactGate>();c->driver=module.driver;c->function=f.value();
  c->grid={p.norm?50U:200U,1,1};
  for(int i=0;i<count;++i)c->pointers[i]=b[i].data;
  *out=std::move(c);return Status::Ok();
}
}
