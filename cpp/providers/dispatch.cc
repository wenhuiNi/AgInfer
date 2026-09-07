#include "providers/dispatch.h"
#include "cublaslt_payload.h"
#include "cuda_kernel_payload.h"
#include <cstring>
#include <utility>

#ifdef AGINFER_HAS_CUDA_PROVIDERS
#include "providers/cublaslt_linear.h"
#include "providers/patch_projection.h"
#include "providers/projection_split.h"
#include "providers/state_update.h"
#include "providers/aot_prefix_input.h"
#include "providers/aot_cast.h"
#include "providers/aot_pointwise.h"
#include "providers/aot_activation.h"
#include "providers/aot_vision_attention.h"
#include "providers/aot_layer_norm.h"
#include "providers/aot_rms_norm.h"
#include "providers/aot_adaptive_rms_norm.h"
#include "providers/aot_kv_pack.h"
#include "providers/aot_prefix_kv_store.h"
#include "providers/aot_suffix_metadata.h"
#include "providers/aot_rope.h"
#include "providers/rounded_attention.h"
#include "providers/aot_time_embedding.h"
#include "providers/aot_action_slice.h"
#ifdef AGINFER_HAS_FLASHINFER
#include "providers/flashinfer_attention.h"
#endif
#endif

namespace aginfer::internal {
namespace {
Status Refused(const char* reason) { return Status(StatusCode::kIncompatibleAbi, reason); }
#ifdef AGINFER_HAS_CUDA_PROVIDERS
template<class Command> class Holder final : public PreparedCommand {
 public:
  std::unique_ptr<Command> command;
  bool library_preflight=false;
  bool NeedsLibraryPreflight() const override { return library_preflight; }
  Status Execute(CudaStream stream) override { return command->Execute(stream); }
};
bool Magic(std::span<const std::uint8_t> payload, const char* magic) {
  return payload.size() >= 8 && std::memcmp(payload.data(), magic, 8) == 0;
}
bool Access(std::span<const CommandBuffer> buffers, const char* pattern) {
  if (buffers.size() != std::strlen(pattern)) return false;
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    auto expected = pattern[i] == 'r' ? CommandOperandAccess::kRead : CommandOperandAccess::kWrite;
    if (buffers[i].access != expected || buffers[i].data == nullptr || buffers[i].bytes == 0) return false;
  }
  return true;
}
template<class Command, class Bindings>
Status Aot(std::span<const std::uint8_t> payload, const Bindings& bindings,
           const CommandModule& module, std::unique_ptr<PreparedCommand>* output,
           bool library_preflight = false) {
  auto holder = std::make_unique<Holder<Command>>();
  holder->library_preflight = library_preflight;
  auto status = Command::Prepare(payload.data(), payload.size(), module.arch, module.bytes,
      module.sha256, module.driver, module.module, bindings, &holder->command);
  if (status.ok()) *output = std::move(holder);
  return status;
}
bool Fits(std::uint64_t a, std::uint64_t b, std::uint64_t scalar, std::uint64_t bytes) {
  return a != 0 && b != 0 && a <= bytes / scalar / b;
}
#endif
}  // namespace

Status PrepareProviderCommand(const CommandRecordView& record,
    std::span<const std::uint8_t> payload, std::span<const CommandBuffer> b,
    const CommandBuffer& workspace, const CommandModule& module,
    std::unique_ptr<PreparedCommand>* output) {
#ifndef AGINFER_HAS_CUDA_PROVIDERS
  (void)record; (void)payload; (void)b; (void)workspace; (void)module; (void)output;
  return Refused("executable provider commands require a CUDA-enabled build");
#else
  if (output == nullptr || record.abi_major != 1 || record.abi_minor != 0)
    return Refused("unsupported executable provider ABI");
  if (record.provider_id == 1 && record.tag == CommandTag::kCublasLtMatmul) {
    if (!Access(b, "rrrw")) return Refused("cuBLASLt operand contract mismatch");
    CublasLtLinearPayloadView p;
    auto status = ParseCublasLtLinearPayload(payload.data(), payload.size(), &p);
    if (!status.ok()) return status;
    const auto scalar = p.dtype == CublasLtDType::kF32 ? 4U : 2U;
    if (p.target_arch != module.arch || !Fits(p.m, p.k, scalar, b[0].bytes) ||
        !Fits(p.n, p.k, scalar, b[1].bytes) || !Fits(1, p.n, scalar, b[2].bytes) ||
        !Fits(p.m, p.n, scalar, b[3].bytes) || workspace.bytes != p.workspace_bytes)
      return Refused("cuBLASLt tensor or workspace range mismatch");
    auto holder = std::make_unique<Holder<CublasLtLinearCommand>>();
    holder->library_preflight = true;
    status = CublasLtLinearCommand::Prepare(payload.data(), payload.size(),
        {b[0].data, b[1].data, b[2].data, b[3].data, workspace.data, workspace.bytes}, &holder->command);
    if (status.ok()) *output = std::move(holder);
    return status;
  }
  if (record.provider_id == 4 && record.tag == CommandTag::kCublasLtMatmul &&
      Magic(payload, "AIPPA1\0")) {
    if (!Access(b, "rrrw")) return Refused("patch projection operand contract mismatch");
    return Aot<PatchProjectionCommand>(payload, PatchProjectionBindings{
        b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes,
        workspace.data,workspace.bytes}, module, output, true);
  }
  if (record.provider_id == 5 && record.tag == CommandTag::kAttention)
    return PrepareRoundedAttention(payload,b,workspace,module,output);
  if (workspace.bytes != 0) return Refused("unexpected workspace for provider command");
#ifdef AGINFER_HAS_FLASHINFER
  if (record.provider_id == 3 && record.tag == CommandTag::kAttention) {
    if (!Access(b, "rrrrw")) return Refused("FlashInfer operand contract mismatch");
    auto holder = std::make_unique<Holder<FlashInferAttentionCommand>>();
    auto status = FlashInferAttentionCommand::Prepare(payload.data(), payload.size(), module.arch,
        {b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes,
         b[4].data,b[4].bytes}, &holder->command);
    if (status.ok()) *output = std::move(holder);
    return status;
  }
#endif
  if (record.provider_id != 2 ||
      record.tag != (Magic(payload, "AIVAT1\0") ? CommandTag::kAttention : CommandTag::kCudaKernel))
    return Refused("no delivered provider for this command ID/tag");
  if (Magic(payload, "AISTU1\0")) {
    if (!Access(b, "rrww")) return Refused("state update operand contract mismatch");
    return PrepareStateUpdate(payload, b, module, output);
  }
  if (Magic(payload, "AIPSP1\0")) {
    if (!Access(b, "rwww")) return Refused("projection split operand contract mismatch");
    return PrepareProjectionSplit(payload, b, module, output);
  }
  if (Magic(payload, "AICUKR1")) {
    CudaKernelPayloadView p;
    auto status = ParseCudaKernelPayload(payload.data(), payload.size(), &p);
    if (!status.ok()) return status;
    auto id = static_cast<std::uint32_t>(p.kernel_id);
    if (id <= 3 && Access(b, "rw"))
      return Aot<AotCastCommand>(payload, AotCastBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes}, module, output);
    if (id >= 4 && id <= 8 && Access(b, "rrw"))
      return Aot<AotPointwiseCommand>(payload, AotPointwiseBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes}, module, output);
    if (id >= 9 && id <= 11 && Access(b, "rw"))
      return Aot<AotActivationCommand>(payload, AotActivationBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes}, module, output);
    return Refused("AOT elementwise operand contract mismatch");
  }
  if (Magic(payload, "AIVAT1\0")) {
    if (!Access(b, "rrrrw")) return Refused("AotVisionAttention operand contract mismatch");
    return Aot<AotVisionAttentionCommand>(payload, AotVisionAttentionBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes,b[4].data,b[4].bytes}, module, output);
  }
  if (Magic(payload, "AILNR1\0")) {
    if (!Access(b, "rrrw")) return Refused("AotLayerNorm operand contract mismatch");
    return Aot<AotLayerNormCommand>(payload, AotLayerNormBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes}, module, output);
  }
  if (Magic(payload, "AIRMS1\0")) {
    if (!Access(b, "rrw")) return Refused("AotRmsNorm operand contract mismatch");
    return Aot<AotRmsNormCommand>(payload, AotRmsNormBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes}, module, output);
  }
  if (Magic(payload, "AIARM1\0")) {
    if (!Access(b, "rrww")) return Refused("AotAdaptiveRmsNorm operand contract mismatch");
    return Aot<AotAdaptiveRmsNormCommand>(payload, AotAdaptiveRmsNormBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes}, module, output);
  }
  if (Magic(payload, "AIKVP1\0")) {
    if (!Access(b, "rrrrww")) return Refused("AotKvPack operand contract mismatch");
    return Aot<AotKvPackCommand>(payload, AotKvPackBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes,b[4].data,b[4].bytes,b[5].data,b[5].bytes}, module, output);
  }
  if (Magic(payload, "AIKVS1\0")) {
    if (!Access(b, "rrww")) return Refused("AotPrefixKvStore operand contract mismatch");
    return Aot<AotPrefixKvStoreCommand>(payload, AotPrefixKvStoreBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes,b[3].data,b[3].bytes}, module, output);
  }
  if (Magic(payload, "AISMD1\0")) {
    if (!Access(b, "rww")) return Refused("AotSuffixMetadata operand contract mismatch");
    return Aot<AotSuffixMetadataCommand>(payload, AotSuffixMetadataBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes}, module, output);
  }
  if (Magic(payload, "AIROP1\0")) {
    if (!Access(b, "rrw")) return Refused("AotRope operand contract mismatch");
    return Aot<AotRopeCommand>(payload, AotRopeBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes,b[2].data,b[2].bytes}, module, output);
  }
  if (Magic(payload, "AITEM1\0")) {
    if (!Access(b, "rw")) return Refused("AotTimeEmbedding operand contract mismatch");
    return Aot<AotTimeEmbeddingCommand>(payload, AotTimeEmbeddingBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes}, module, output);
  }
  if (Magic(payload, "AISLC1\0")) {
    if (!Access(b, "rw")) return Refused("AotActionSlice operand contract mismatch");
    return Aot<AotActionSliceCommand>(payload, AotActionSliceBindings{b[0].data,b[0].bytes,b[1].data,b[1].bytes}, module, output);
  }
  if (Magic(payload, "AIPIA1\0")) {
    if (!Access(b, "rrrrrrrrrwww")) return Refused("prefix input operand contract mismatch");
    AotPrefixInputBindings bindings;
    for (int i = 0; i < 3; ++i) {
      bindings.images[i] = b[i].data; bindings.image_bytes[i] = b[i].bytes;
      bindings.image_masks[i] = b[5+i].data; bindings.image_mask_bytes[i] = b[5+i].bytes;
    }
    bindings.embedding = b[3].data; bindings.embedding_bytes = b[3].bytes;
    bindings.tokens = b[4].data; bindings.token_bytes = b[4].bytes;
    bindings.token_mask = b[8].data; bindings.token_mask_bytes = b[8].bytes;
    bindings.prefix = b[9].data; bindings.prefix_bytes = b[9].bytes;
    bindings.pad_mask = b[10].data; bindings.pad_mask_bytes = b[10].bytes;
    bindings.positions = b[11].data; bindings.position_bytes = b[11].bytes;
    return Aot<AotPrefixInputCommand>(payload, bindings, module, output);
  }
  return Refused("no delivered AOT executable form for payload");
#endif
}
}  // namespace aginfer::internal
