#pragma once
#include "providers/dispatch.h"
namespace aginfer::internal {
Status PrepareRoundedMulAdd(std::span<const std::uint8_t>, std::span<const CommandBuffer>,
                            const CommandModule&, std::unique_ptr<PreparedCommand>*);
}
