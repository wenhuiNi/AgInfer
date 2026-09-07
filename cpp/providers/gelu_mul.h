#pragma once
#include "providers/dispatch.h"
namespace aginfer::internal {
Status PrepareGeluMul(std::span<const std::uint8_t>, std::span<const CommandBuffer>,
                      const CommandModule&, std::unique_ptr<PreparedCommand>*);
}
