#pragma once
#include "providers/dispatch.h"
namespace aginfer::internal {
Status PrepareRoundedAttention(std::span<const std::uint8_t> payload,
    std::span<const CommandBuffer> buffers,const CommandBuffer& workspace,const CommandModule& module,
    std::unique_ptr<PreparedCommand>* output);
}
