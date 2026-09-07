#include "providers/rounded_mul_add.h"
#include "rounded_mul_add_payload.h"
#include <algorithm>
namespace aginfer::internal {
namespace {
class RoundedMulAdd final : public PreparedCommand {
 public:
  CudaDriver* driver=nullptr;
  CuFunction function=nullptr;
  std::array<void*,4> pointers{};
  std::uint64_t numel=0;
  std::array<std::uint32_t,3> grid{},block{256,1,1};
  Status Execute(CudaStream stream) override {
    void* args[]={&pointers[0],&pointers[1],&pointers[2],&pointers[3],&numel};
    return driver->Launch(function,grid.data(),block.data(),0,stream,args);
  }
};
}
Status PrepareRoundedMulAdd(std::span<const std::uint8_t> payload, std::span<const CommandBuffer> b,
                            const CommandModule& module, std::unique_ptr<PreparedCommand>* out) {
  auto bad=[] {return Status(StatusCode::kInvalidArgument,"rounded multiply-add binding mismatch");};
  if(!out)return bad();
  out->reset();
  RoundedMulAddPayloadView p;
  auto status=ParseRoundedMulAddPayload(payload.data(),payload.size(),&p);
  if(!status.ok())return status;
  if(module.arch!=120 || module.bytes!=p.module_bytes || module.sha256!=p.module_sha256
      || !module.driver || !module.module || b.size()!=4)return bad();
  auto bytes=p.numel*2;
  for(int i=0;i<4;++i) {
    if(!b[i].data || reinterpret_cast<std::uintptr_t>(b[i].data)%2 || b[i].bytes<bytes
        || b[i].access!=(i==3?CommandOperandAccess::kWrite:CommandOperandAccess::kRead))return bad();
  }
  auto output=reinterpret_cast<std::uintptr_t>(b[3].data);
  for(int i=0;i<3;++i) {
    auto input=reinterpret_cast<std::uintptr_t>(b[i].data);
    if(output<=input?input-output<bytes:output-input<bytes)return bad();
  }
  auto function=module.driver->GetFunction(module.module,"aginfer_rounded_mul_add_bf16");
  if(!function.ok())return function.status();
  auto c=std::make_unique<RoundedMulAdd>();
  c->driver=module.driver;c->function=function.value();c->numel=p.numel;
  for(int i=0;i<4;++i)c->pointers[i]=b[i].data;
  c->grid={static_cast<std::uint32_t>(std::min<std::uint64_t>((p.numel+255)/256,4096)),1,1};
  *out=std::move(c);return Status::Ok();
}
}
