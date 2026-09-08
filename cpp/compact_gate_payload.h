#pragma once
#include "rounded_mul_add_payload.h"
namespace aginfer::internal {
struct CompactGatePayloadView : RoundedMulAddPayloadView { bool norm = false; };
Status ParseCompactGatePayload(const std::uint8_t*, std::size_t, CompactGatePayloadView*);
}
