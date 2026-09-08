#include "compact_gate_payload.h"
#include <array>
#include <cstring>
namespace aginfer::internal {
Status ParseCompactGatePayload(const std::uint8_t* data, std::size_t size, CompactGatePayloadView* out) {
  auto bad=[] {return Status(StatusCode::kCorruptPackage,"invalid compact-gate payload");};
  if(!data || !out || size!=128)return bad();
  const bool norm=std::memcmp(data,"AIANG1\0",8)==0;
  if(!norm && std::memcmp(data,"AIGRD1\0",8))return bad();
  std::array<std::uint8_t,128> base;
  std::memcpy(base.data(),data,size);std::memcpy(base.data(),"AIMAD1\0",8);
  CompactGatePayloadView p;
  auto status=ParseRoundedMulAddPayload(base.data(),size,&p);
  if(!status.ok())return status;
  if(p.numel!=51200)return bad();
  p.norm=norm;*out=p;return Status::Ok();
}
}
