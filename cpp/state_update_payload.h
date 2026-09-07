#pragma once
#include "status.h"
#include <array>
#include <cstddef>
#include <cstdint>
namespace aginfer::internal {
struct StateUpdatePayloadView {
  std::uint64_t total=0,count=0,offset=0,module_bytes=0;
  std::array<std::uint8_t,32> module_sha256{};
};
Status ParseStateUpdatePayload(const std::uint8_t*,std::size_t,StateUpdatePayloadView*);
}
