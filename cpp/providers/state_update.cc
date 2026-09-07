#include "providers/state_update.h"
#include "state_update_payload.h"
namespace aginfer::internal {
namespace {
class StateUpdate final:public PreparedCommand {
 public:
  CudaDriver* driver=nullptr; CuFunction function=nullptr;
  std::array<void*,4> pointers{};
  std::uint64_t vectors=0,offset=0;
  std::array<std::uint32_t,3> grid{},block{256,1,1};
  Status Execute(CudaStream stream) override {
    void* args[]={&pointers[0],&pointers[1],&pointers[2],&pointers[3],&vectors,&offset};
    return driver->Launch(function,grid.data(),block.data(),0,stream,args);
  }
};
}
Status PrepareStateUpdate(std::span<const std::uint8_t> payload,std::span<const CommandBuffer> b,
                          const CommandModule& module,std::unique_ptr<PreparedCommand>* out) {
  auto bad=[] {return Status(StatusCode::kInvalidArgument,"state update binding mismatch");};
  if(!out)return bad();
  StateUpdatePayloadView p;auto status=ParseStateUpdatePayload(payload.data(),payload.size(),&p);
  if(!status.ok())return status;
  if(module.arch!=120||module.bytes!=p.module_bytes||module.sha256!=p.module_sha256
      ||!module.driver||!module.module||b.size()!=4)return bad();
  for(int i=0;i<4;++i) {
    const auto bytes=(i<2?p.count:p.total)*2;
    if(!b[i].data||reinterpret_cast<std::uintptr_t>(b[i].data)%16||b[i].bytes<bytes
        ||b[i].access!=(i<2?CommandOperandAccess::kRead:CommandOperandAccess::kWrite))return bad();
    for(int j=0;j<i;++j) {
      // Source/source overlap is harmless; source/destination and cache/cache
      // overlap would violate the fixed copy contract.
      if(i<2)continue;
      auto x=reinterpret_cast<std::uintptr_t>(b[i].data),y=reinterpret_cast<std::uintptr_t>(b[j].data);
      if(x<=y?y-x<bytes:x-y<(j<2?p.count:p.total)*2)return bad();
    }
  }
  auto function=module.driver->GetFunction(module.module,"aginfer_state_update_bf16_pair");
  if(!function.ok())return function.status();
  auto cmd=std::make_unique<StateUpdate>();cmd->driver=module.driver;cmd->function=function.value();
  for(int i=0;i<4;++i)cmd->pointers[i]=b[i].data;
  cmd->vectors=p.count/8;cmd->offset=p.offset/8;
  cmd->grid={static_cast<std::uint32_t>((2*cmd->vectors+255)/256),1,1};
  *out=std::move(cmd);return Status::Ok();
}
}
