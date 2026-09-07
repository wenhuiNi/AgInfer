// Offline compiler tool only. No weights, inference, timing or runtime search.
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void Check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLASLt status " + std::to_string(status));
}
struct Layout { std::int64_t rows, cols, ld, stride; };
struct Request {
  int dtype, compute, bias, batch, trans_a, trans_b;
  std::array<Layout, 3> layouts;
  std::uint64_t workspace;
  std::array<std::uint32_t, 3> alignments;
};
std::vector<Request> ReadRequests() {
  std::string input;
  char ch;
  while (std::cin.get(ch)) {
    Require(input.size() < 65536, "request exceeds 64 KiB");
    input += ch;
  }
  std::istringstream stream(input);
  std::string magic;
  int count = 0;
  stream >> magic >> count;
  Require(magic == "AGINFER_LT_SELECT_V1" && count > 0 && count <= 256,
          "unsupported selection protocol or count");
  std::vector<Request> requests;
  for (int index = 0; index < count; ++index) {
    Request r{};
    stream >> r.dtype >> r.compute >> r.bias >> r.batch >> r.trans_a >> r.trans_b;
    for (auto& x : r.layouts) stream >> x.rows >> x.cols >> x.ld >> x.stride;
    stream >> r.workspace;
    for (auto& x : r.alignments) stream >> x;
    Require(bool(stream), "truncated or nonnumeric selection request");
    Require((r.dtype == 1 || r.dtype == 2) && (r.compute == 1 || r.compute == 2) &&
            (r.compute == 1 || r.dtype == 1) && (r.bias == 0 || r.bias == 1) &&
            r.batch > 0 && r.batch <= 64 && (r.trans_a == 0 || r.trans_a == 1) &&
            (r.trans_b == 0 || r.trans_b == 1) && r.workspace <= (64ULL << 20),
            "unsupported dtype/compute/batch/workspace");
    for (const auto& x : r.layouts) {
      Require(x.rows > 0 && x.rows <= 32768 && x.cols > 0 && x.cols <= 32768 &&
              x.ld >= x.rows && x.ld <= (1 << 20) && x.stride >= 0 &&
              x.stride <= (1LL << 30), "invalid matrix layout");
      Require((x.cols - 1) * x.ld + x.rows + (r.batch - 1) * x.stride <= (1LL << 32),
              "matrix span exceeds selection envelope");
    }
    const auto& a = r.layouts[0]; const auto& b = r.layouts[1]; const auto& c = r.layouts[2];
    Require((r.trans_a ? a.cols : a.rows) == c.rows &&
            (r.trans_b ? b.rows : b.cols) == c.cols &&
            (r.trans_a ? a.rows : a.cols) == (r.trans_b ? b.cols : b.rows) &&
            (r.batch == 1 || c.stride > 0), "matrix multiplication dimensions do not close");
    for (auto x : r.alignments)
      Require(x > 0 && x <= 256 && !(x & (x - 1)), "invalid pointer alignment");
    requests.push_back(r);
  }
  stream >> std::ws;
  Require(stream.eof(), "unexpected trailing request data");
  return requests;
}
struct Resources {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  std::array<cublasLtMatrixLayout_t, 3> layouts{};
  ~Resources() {
    for (auto x : layouts) if (x) cublasLtMatrixLayoutDestroy(x);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (op) cublasLtMatmulDescDestroy(op);
    if (handle) cublasLtDestroy(handle);
  }
};
constexpr std::array attributes{
    CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
bool Config(const cublasLtMatmulAlgo_t& algorithm, std::array<std::uint32_t, 9>& out) {
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::size_t written = 0;
    const auto bytes = i >= 7 ? sizeof(std::uint16_t) : sizeof(std::uint32_t);
    out[i] = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attributes[i], &out[i], bytes,
                                          &written) != CUBLAS_STATUS_SUCCESS || written != bytes ||
        out[i] >= (1U << 31)) return false;
  }
  return true;
}
std::string Select(const Request& r) {
  Resources ctx;
  Check(cublasLtCreate(&ctx.handle));
  const auto dtype = r.dtype == 1 ? CUDA_R_32F : CUDA_R_16BF;
  const auto compute = r.compute == 1 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
  Check(cublasLtMatmulDescCreate(&ctx.op, compute, CUDA_R_32F));
  const cublasOperation_t ta = r.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t tb = r.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto epilogue = r.bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
  Check(cublasLtMatmulDescSetAttribute(ctx.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  Check(cublasLtMatmulDescSetAttribute(ctx.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  Check(cublasLtMatmulDescSetAttribute(ctx.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  for (std::size_t i = 0; i < ctx.layouts.size(); ++i) {
    const auto& x = r.layouts[i];
    Check(cublasLtMatrixLayoutCreate(&ctx.layouts[i], dtype, x.rows, x.cols, x.ld));
    if (r.batch > 1) {
      Check(cublasLtMatrixLayoutSetAttribute(ctx.layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                           &r.batch, sizeof(r.batch)));
      Check(cublasLtMatrixLayoutSetAttribute(ctx.layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                           &x.stride, sizeof(x.stride)));
    }
  }
  Check(cublasLtMatmulPreferenceCreate(&ctx.preference));
  Check(cublasLtMatmulPreferenceSetAttribute(ctx.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &r.workspace, sizeof(r.workspace)));
  constexpr std::array prefs{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
  for (std::size_t i = 0; i < prefs.size(); ++i) {
    const auto alignment = r.alignments[i == 3 ? 2 : i];
    Check(cublasLtMatmulPreferenceSetAttribute(ctx.preference, prefs[i], &alignment, sizeof(alignment)));
  }
  std::array<cublasLtMatmulHeuristicResult_t, 32> candidates{};
  int count = 0;
  Check(cublasLtMatmulAlgoGetHeuristic(ctx.handle, ctx.op, ctx.layouts[0], ctx.layouts[1],
      ctx.layouts[2], ctx.layouts[2], ctx.preference, candidates.size(), candidates.data(), &count));
  for (int index = 0; index < count; ++index) {
    if (candidates[index].state != CUBLAS_STATUS_SUCCESS) continue;
    std::array<std::uint32_t, 9> config{};
    if (!Config(candidates[index].algo, config)) continue;
    cublasLtMatmulAlgo_t reconstructed{};
    if (cublasLtMatmulAlgoInit(ctx.handle, compute, CUDA_R_32F, dtype, dtype, dtype, dtype,
                              config[0], &reconstructed) != CUBLAS_STATUS_SUCCESS) continue;
    bool valid = true;
    for (std::size_t i = 1; i < config.size(); ++i) {
      const auto bytes = i >= 7 ? sizeof(std::uint16_t) : sizeof(std::uint32_t);
      valid &= cublasLtMatmulAlgoConfigSetAttribute(&reconstructed, attributes[i], &config[i], bytes)
               == CUBLAS_STATUS_SUCCESS;
    }
    std::array<std::uint32_t, 9> roundtrip{};
    if (!valid || !Config(reconstructed, roundtrip) || roundtrip != config) continue;
    cublasLtMatmulHeuristicResult_t checked{};
    if (cublasLtMatmulAlgoCheck(ctx.handle, ctx.op, ctx.layouts[0], ctx.layouts[1],
        ctx.layouts[2], ctx.layouts[2], &reconstructed, &checked) != CUBLAS_STATUS_SUCCESS ||
        checked.state != CUBLAS_STATUS_SUCCESS || checked.workspaceSize > r.workspace) continue;
    std::ostringstream out;
    out << "{\"algorithm\":[";
    for (std::size_t i = 0; i < config.size(); ++i) out << (i ? "," : "") << config[i];
    out << "],\"workspace_bytes\":" << checked.workspaceSize << ",\"heuristic_rank\":" << index
        << ",\"candidate_count\":" << count << ",\"algo_check\":true}";
    return out.str();
  }
  throw std::runtime_error("no reconstructable AlgoCheck candidate within declared envelope");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--check-input"), "unknown option");
    const auto requests = ReadRequests();
    if (argc == 2) {
      std::cout << "{\"request_count\":" << requests.size() << ",\"input_valid\":true}\n";
      return 0;
    }
    int device = 0, driver = 0;
    cudaDeviceProp prop{};
    Require(cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
            cudaDriverGetVersion(&driver) == cudaSuccess, "CUDA device unavailable");
    Require(prop.major * 10 + prop.minor == 120 && cublasLtGetVersion() == 120803,
            "selector requires exact SM120 and cuBLASLt 120803");
    std::ostringstream out;
    out << "{\"schema\":\"aginfer.lt-selection-probe.v1\",\"arch\":120,\"cublaslt_version\":120803,"
        << "\"cuda_driver_version\":" << driver << ",\"results\":[";
    for (std::size_t i = 0; i < requests.size(); ++i) {
      try { out << (i ? "," : "") << Select(requests[i]); }
      catch (const std::exception& e) { throw std::runtime_error("request " + std::to_string(i) + ": " + e.what()); }
    }
    std::cout << out.str() << "]}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "selection refused: " << e.what() << '\n';
    return 2;
  }
}
