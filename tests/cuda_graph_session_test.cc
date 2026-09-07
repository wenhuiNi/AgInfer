#include "aginfer/c_api.h"
#include <cuda_runtime_api.h>
#include <array>
#include <iostream>
#include <stdexcept>

void CheckAt(bool value, int line) {
  if (!value) throw std::runtime_error("CUDA graph contract failed at line " + std::to_string(line));
}
#define Check(value) CheckAt((value), __LINE__)
void Ai(ai_status status) { if (status) throw std::runtime_error(ai_last_error()); }
void Cu(cudaError_t status) { if (status) throw std::runtime_error(cudaGetErrorString(status)); }
struct Session {
  ai_session* session = nullptr;
  std::array<float*, 3> buffers{};
  Session(ai_runtime* runtime, ai_model* model, bool explicit_options) {
    ai_session_options options; ai_session_options_init(&options);
    Ai(ai_session_create(runtime, model, explicit_options ? &options : nullptr, &session));
    for (unsigned p = 0; p < 3; ++p) {
      Cu(cudaMalloc(reinterpret_cast<void**>(&buffers[p]), 1024));
      Ai(Bind(p, buffers[p]));
    }
  }
  ai_status Bind(unsigned p, float* address) {
    ai_tensor_view view; ai_tensor_view_init(&view);
    const int64_t shape = 256, stride = 1;
    view.dtype = AI_DTYPE_F32; view.location = AI_MEMORY_DEVICE; view.rank = 1;
    view.shape = &shape; view.stride = &stride; view.byte_size = 1024; view.data = address;
    return p == 2 ? ai_session_bind_output(session, p, &view) : ai_session_bind_input(session, p, &view);
  }
  void Fill(unsigned p, float value) {
    std::array<float, 256> data; data.fill(value);
    Cu(cudaMemcpy(buffers[p], data.data(), 1024, cudaMemcpyHostToDevice));
    // Pageable H2D may return after staging, before the device copy completes.
    // Run uses a nonblocking stream; explicitly finish input upload first.
    Cu(cudaStreamSynchronize(nullptr));
  }
  void Output(float expected) {
    std::array<float, 256> data;
    Cu(cudaMemcpy(data.data(), buffers[2], 1024, cudaMemcpyDeviceToHost));
    for (float value : data)
      if (value != expected) throw std::runtime_error("output " + std::to_string(value) +
          " differs from " + std::to_string(expected));
  }
  void Run(cudaStream_t stream, float expected) {
    Ai(ai_session_enqueue(session, stream)); Cu(cudaStreamSynchronize(stream)); Output(expected);
  }
  void Ledger(uint64_t runs, bool graph) {
    ai_execution_info info; ai_execution_info_init(&info);
    for (unsigned p : {0, 2}) {
      Ai(ai_session_get_execution_info(session, p, &info));
      Check(info.enqueues == runs && info.commands_per_enqueue == 2 &&
          info.commands_submitted == 2 * runs && !info.fallback_count);
    }
    ai_cuda_graph_info capture; ai_cuda_graph_info_init(&capture);
    Ai(ai_session_get_cuda_graph_info(session, &capture));
    Check(capture.enabled == graph && capture.instantiated == graph &&
        capture.node_count == (graph ? 2u : 0u) && capture.launches == (graph ? runs : 0u));
  }
  ~Session() { ai_session_destroy(session); for (auto buffer : buffers) cudaFree(buffer); }
};
int main(int argc, char** argv) { try {
  if (argc != 3) return 2;
  ai_runtime* runtime; ai_model *model, *bad;
  Ai(ai_runtime_create(nullptr, &runtime)); Ai(ai_model_load(argv[1], &model));
  Ai(ai_model_load(argv[2], &bad));
  cudaStream_t stream; Cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    Session defaults(runtime, model, false), graph(runtime, model, true), peer(runtime, model, true);
    for (auto arm : {&defaults, &graph, &peer}) {
      arm->Fill(0, 3); arm->Fill(1, 5); arm->Fill(2, -123);
      Ai(ai_session_prepare(arm->session)); arm->Output(-123);
      Ai(ai_session_prepare(arm->session)); arm->Output(-123);
      Check(arm->Bind(0, arm->buffers[1]) == AI_STATUS_INVALID_STATE);
      Ai(arm->Bind(0, arm->buffers[0]));
    }
    defaults.Ledger(0, true); graph.Ledger(0, true); peer.Ledger(0, true);
    defaults.Run(stream, 13); graph.Run(stream, 13); graph.Run(nullptr, 13);
    graph.Fill(0, -7); defaults.Fill(0, -7);
    defaults.Run(stream, 3); graph.Run(stream, 3); peer.Run(stream, 13);
    graph.Fill(0, 3); graph.Run(stream, 13);
    Cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    Check(ai_session_enqueue(graph.session, stream) == AI_STATUS_INVALID_STATE);
    cudaGraph_t outer; Cu(cudaStreamEndCapture(stream, &outer)); Cu(cudaGraphDestroy(outer));
    graph.Ledger(4, true); defaults.Ledger(2, true); peer.Ledger(1, true);
    graph.Run(stream, 13); graph.Ledger(5, true);
    Session invalid(runtime, bad, true); invalid.Fill(2, -123);
    Check(ai_session_prepare(invalid.session) != AI_STATUS_OK); invalid.Output(-123);
    Check(ai_session_prepare(invalid.session) == AI_STATUS_INVALID_STATE);
    Check(ai_session_enqueue(invalid.session, stream) == AI_STATUS_INVALID_STATE);
    ai_cuda_graph_info failed; ai_cuda_graph_info_init(&failed);
    Ai(ai_session_get_cuda_graph_info(invalid.session, &failed));
    Check(failed.enabled && !failed.instantiated && !failed.node_count && !failed.launches);
  }
  Cu(cudaStreamDestroy(stream)); ai_model_destroy(bad); ai_model_destroy(model); ai_runtime_destroy(runtime);
  std::cout << "default/zero-option graph replay, refresh/isolation, prepare poison, ledger and refusal contracts passed\n";
  return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
