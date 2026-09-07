#include "providers/cublaslt_linear.h"

#include "cublaslt_payload.h"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace aginfer::internal {
namespace {

Status Invalid(const std::string& message) {
  return Status(StatusCode::kInvalidArgument, message);
}

Status IncompatibleArch(const std::string& message) {
  return Status(StatusCode::kIncompatibleArchitecture, message);
}

Status IncompatibleAbi(const std::string& message) {
  return Status(StatusCode::kIncompatibleAbi, message);
}

Status CudaError(const std::string& operation, cudaError_t status) {
  return Status(StatusCode::kCudaError,
                operation + ": " + cudaGetErrorString(status));
}

Status CublasError(const std::string& operation, cublasStatus_t status) {
  return Status(StatusCode::kCudaError,
                operation + ": cuBLAS status " +
                    std::to_string(static_cast<int>(status)));
}

bool Aligned(const void* pointer, std::uint32_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

template <typename T>
Status SetAlgorithmConfig(cublasLtMatmulAlgo_t* algorithm,
                          cublasLtMatmulAlgoConfigAttributes_t attribute,
                          const T& value, const char* label) {
  const cublasStatus_t status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, attribute, &value, sizeof(value));
  return status == CUBLAS_STATUS_SUCCESS ? Status::Ok()
                                         : CublasError(label, status);
}

}  // namespace

struct CublasLtLinearCommand::Impl {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  CublasLtLinearBindings bindings;
  std::size_t workspace_bytes = 0;

  ~Impl() {
    if (layout_d != nullptr) cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != nullptr) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != nullptr) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != nullptr) cublasLtMatrixLayoutDestroy(layout_a);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (handle != nullptr) cublasLtDestroy(handle);
  }
};

CublasLtLinearCommand::CublasLtLinearCommand(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CublasLtLinearCommand::~CublasLtLinearCommand() = default;

Status CublasLtLinearCommand::Prepare(
    const std::uint8_t* payload, std::size_t payload_size,
    const CublasLtLinearBindings& bindings,
    std::unique_ptr<CublasLtLinearCommand>* output) {
  if (output == nullptr) return Invalid("cuBLASLt command output is null");
  output->reset();
  CublasLtLinearPayloadView parsed;
  Status status = ParseCublasLtLinearPayload(payload, payload_size, &parsed);
  if (!status.ok()) return status;
  if (bindings.x == nullptr || bindings.weight == nullptr ||
      bindings.bias == nullptr || bindings.output == nullptr) {
    return Invalid("cuBLASLt command has a null tensor binding");
  }
  if (!Aligned(bindings.x, parsed.x_alignment) ||
      !Aligned(bindings.weight, parsed.weight_alignment) ||
      !Aligned(bindings.bias, parsed.bias_alignment) ||
      !Aligned(bindings.output, parsed.output_alignment)) {
    return Invalid("cuBLASLt command tensor binding does not satisfy payload alignment");
  }
  if (parsed.workspace_bytes == 0) {
    if (bindings.workspace != nullptr || bindings.workspace_bytes != 0) {
      return Invalid("zero-workspace cuBLASLt command received a workspace binding");
    }
  } else if (bindings.workspace == nullptr ||
             !Aligned(bindings.workspace, parsed.workspace_alignment) ||
             bindings.workspace_bytes < parsed.workspace_bytes) {
    return Invalid("cuBLASLt command workspace binding is missing, short, or misaligned");
  }
  if (parsed.workspace_bytes > std::numeric_limits<std::size_t>::max()) {
    return Invalid("cuBLASLt command workspace does not fit size_t");
  }

  int device = 0;
  cudaError_t cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) return CudaError("cudaGetDevice", cuda_status);
  cudaDeviceProp properties{};
  cuda_status = cudaGetDeviceProperties(&properties, device);
  if (cuda_status != cudaSuccess) {
    return CudaError("cudaGetDeviceProperties", cuda_status);
  }
  const std::uint32_t active_arch =
      static_cast<std::uint32_t>(properties.major * 10 + properties.minor);
  if (active_arch != parsed.target_arch) {
    return IncompatibleArch("cuBLASLt payload target does not match the active device");
  }
  if (cublasLtGetVersion() != parsed.cublaslt_version) {
    return IncompatibleAbi("cuBLASLt payload library version does not match runtime");
  }

  auto impl = std::make_unique<Impl>();
  impl->bindings = bindings;
  impl->workspace_bytes = static_cast<std::size_t>(parsed.workspace_bytes);
  cublasStatus_t cublas_status = cublasLtCreate(&impl->handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cublasLtCreate", cublas_status);
  }
  const cudaDataType_t dtype = parsed.dtype == CublasLtDType::kF32
                                  ? CUDA_R_32F
                                  : CUDA_R_16BF;
  const auto compute=parsed.compute_mode==2 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  cublas_status = cublasLtMatmulDescCreate(
      &impl->operation, compute, CUDA_R_32F);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cublasLtMatmulDescCreate", cublas_status);
  }
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  for (const auto& setting : {
           std::pair{CUBLASLT_MATMUL_DESC_TRANSA,
                     std::pair{static_cast<const void*>(&trans_a), sizeof(trans_a)}},
           std::pair{CUBLASLT_MATMUL_DESC_TRANSB,
                     std::pair{static_cast<const void*>(&trans_b), sizeof(trans_b)}},
           std::pair{CUBLASLT_MATMUL_DESC_EPILOGUE,
                     std::pair{static_cast<const void*>(&epilogue), sizeof(epilogue)}},
           std::pair{CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                     std::pair{static_cast<const void*>(&impl->bindings.bias),
                               sizeof(impl->bindings.bias)}},
       }) {
    cublas_status = cublasLtMatmulDescSetAttribute(
        impl->operation, setting.first, setting.second.first,
        setting.second.second);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return CublasError("cublasLtMatmulDescSetAttribute", cublas_status);
    }
  }

  for (const auto& layout : {
           std::pair{&impl->layout_a,
                     std::pair{std::pair{parsed.k, parsed.n}, parsed.lda}},
           std::pair{&impl->layout_b,
                     std::pair{std::pair{parsed.k, parsed.m}, parsed.ldb}},
           std::pair{&impl->layout_c,
                     std::pair{std::pair{parsed.n, parsed.m}, parsed.ldc}},
           std::pair{&impl->layout_d,
                     std::pair{std::pair{parsed.n, parsed.m}, parsed.ldd}},
       }) {
    cublas_status = cublasLtMatrixLayoutCreate(
        layout.first, dtype, layout.second.first.first,
        layout.second.first.second, layout.second.second);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return CublasError("cublasLtMatrixLayoutCreate", cublas_status);
    }
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
    cublas_status = cublasLtMatrixLayoutSetAttribute(
        *layout.first, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      return CublasError("cublasLtMatrixLayoutSetAttribute", cublas_status);
    }
  }

  cublas_status = cublasLtMatmulAlgoInit(
      impl->handle, compute, CUDA_R_32F, dtype, dtype, dtype,
      dtype, parsed.algorithm.algorithm_id, &impl->algorithm);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cublasLtMatmulAlgoInit", cublas_status);
  }
  for (Status config_status : {
           SetAlgorithmConfig(&impl->algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID,
                              parsed.algorithm.tile_id, "set tile ID"),
           SetAlgorithmConfig(&impl->algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                              parsed.algorithm.split_k, "set split-K"),
           SetAlgorithmConfig(&impl->algorithm,
                              CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                              parsed.algorithm.reduction_scheme,
                              "set reduction scheme"),
           SetAlgorithmConfig(&impl->algorithm,
                              CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                              parsed.algorithm.cta_swizzling,
                              "set CTA swizzling"),
           SetAlgorithmConfig(&impl->algorithm,
                              CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                              parsed.algorithm.custom_option,
                              "set custom option"),
           SetAlgorithmConfig(&impl->algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                              parsed.algorithm.stages_id, "set stages ID"),
           SetAlgorithmConfig(&impl->algorithm,
                              CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
                              parsed.algorithm.inner_shape_id,
                              "set inner-shape ID"),
           SetAlgorithmConfig(&impl->algorithm,
                              CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                              parsed.algorithm.cluster_shape_id,
                              "set cluster-shape ID"),
       }) {
    if (!config_status.ok()) return config_status;
  }
  cublasLtMatmulHeuristicResult_t checked{};
  cublas_status = cublasLtMatmulAlgoCheck(
      impl->handle, impl->operation, impl->layout_a, impl->layout_b,
      impl->layout_c, impl->layout_d, &impl->algorithm, &checked);
  if (cublas_status != CUBLAS_STATUS_SUCCESS ||
      checked.state != CUBLAS_STATUS_SUCCESS) {
    return CublasError("cublasLtMatmulAlgoCheck",
                       cublas_status != CUBLAS_STATUS_SUCCESS
                           ? cublas_status
                           : checked.state);
  }
  if (checked.workspaceSize != parsed.workspace_bytes) {
    return IncompatibleAbi(
        "cuBLASLt reconstructed algorithm workspace differs from payload");
  }
  output->reset(new CublasLtLinearCommand(std::move(impl)));
  return Status::Ok();
}

Status CublasLtLinearCommand::Execute(void* cuda_stream) {
  if (impl_ == nullptr) return Invalid("cuBLASLt command is not prepared");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const cublasStatus_t status = cublasLtMatmul(
      impl_->handle, impl_->operation, &alpha, impl_->bindings.weight,
      impl_->layout_a, impl_->bindings.x, impl_->layout_b, &beta,
      impl_->bindings.output, impl_->layout_c, impl_->bindings.output,
      impl_->layout_d, &impl_->algorithm, impl_->bindings.workspace,
      impl_->workspace_bytes, static_cast<cudaStream_t>(cuda_stream));
  return status == CUBLAS_STATUS_SUCCESS
             ? Status::Ok()
             : CublasError("cublasLtMatmul", status);
}

}  // namespace aginfer::internal
