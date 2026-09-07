// Offline compiler helper. Executes only a compiler-built zero-input AIM,
// preserving its exact native payloads and graph policy. Not a runtime feature.
#include "aginfer/c_api.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void Ai(ai_status status) { if (status) throw std::runtime_error(ai_last_error()); }
void Cu(cudaError_t status) { if (status) throw std::runtime_error(cudaGetErrorString(status)); }
struct Resources {
  ai_runtime* runtime = nullptr;
  ai_model* model = nullptr;
  ai_session* session = nullptr;
  cudaStream_t stream = nullptr;
  std::vector<void*> outputs;
  ~Resources() {
    if (stream) cudaStreamSynchronize(stream);
    ai_session_destroy(session); ai_model_destroy(model); ai_runtime_destroy(runtime);
    for (void* p : outputs) cudaFree(p);
    if (stream) cudaStreamDestroy(stream);
  }
};
}
int main(int argc, char** argv) { try {
  Require(argc == 3, "usage: aginfer_constant_eval evaluation.aim output.bin");
  Require(!std::filesystem::exists(argv[2]), "constant evaluator output already exists");
  Require(std::filesystem::file_size(argv[1]) <= (2ULL << 30), "evaluation AIM exceeds 2 GiB");
  Resources r;
  Ai(ai_runtime_create(nullptr, &r.runtime)); Ai(ai_model_load(argv[1], &r.model));
  Ai(ai_session_create(r.runtime, r.model, nullptr, &r.session));
  unsigned inputs, outputs;
  Ai(ai_session_get_port_count(r.session, AI_PORT_INPUT, &inputs));
  Ai(ai_session_get_port_count(r.session, AI_PORT_OUTPUT, &outputs));
  Require(inputs == 0 && outputs > 0 && outputs <= 4096, "evaluation must have only bounded constant outputs");
  std::vector<std::size_t> sizes;
  std::size_t total = 0;
  for (unsigned i = 0; i < outputs; ++i) {
    ai_port_info port; ai_port_info_init(&port);
    Ai(ai_session_get_port_info(r.session, AI_PORT_OUTPUT, i, &port));
    Require(port.dtype == AI_DTYPE_F32 && port.byte_size && port.byte_size % 4 == 0 &&
        port.byte_size <= (16U << 20) - total, "evaluation output exceeds F32 byte budget");
    total += port.byte_size; sizes.push_back(port.byte_size);
    void* ptr = nullptr; Cu(cudaMalloc(&ptr, port.byte_size)); r.outputs.push_back(ptr);
    ai_tensor_view view; ai_tensor_view_init(&view);
    view.dtype = port.dtype; view.location = port.location; view.rank = port.rank;
    view.shape = port.shape; view.stride = port.stride; view.byte_size = port.byte_size; view.data = ptr;
    Ai(ai_session_bind_output(r.session, port.port_id, &view));
  }
  ai_execution_info execution; ai_execution_info_init(&execution);
  Ai(ai_session_get_execution_info(r.session, 0, &execution));
  Require(execution.commands_per_enqueue > 0 && execution.commands_per_enqueue <= 4096,
      "constant evaluator command budget exceeded");
  Ai(ai_session_prepare(r.session));
  Cu(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
  std::vector<char> reference(total), result(total);
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (unsigned i = 0; i < outputs; ++i) Cu(cudaMemsetAsync(r.outputs[i], 0xff, sizes[i], r.stream));
    Ai(ai_session_enqueue(r.session, r.stream)); Cu(cudaStreamSynchronize(r.stream));
    std::size_t offset = 0;
    for (unsigned i = 0; i < outputs; ++i) {
      Cu(cudaMemcpy(result.data() + offset, r.outputs[i], sizes[i], cudaMemcpyDeviceToHost));
      offset += sizes[i];
    }
    for (std::size_t i = 0; i < total; i += 4) {
      float value; std::memcpy(&value, result.data() + i, 4);
      Require(std::isfinite(value), "constant evaluation produced non-finite output");
    }
    if (repeat == 0) reference = result;
    else Require(result == reference, "constant evaluator graph repeat differs");
  }
  ai_cuda_graph_info graph; ai_cuda_graph_info_init(&graph);
  Ai(ai_session_get_cuda_graph_info(r.session, &graph));
  Ai(ai_session_get_execution_info(r.session, 0, &execution));
  Require(graph.instantiated && graph.node_count && graph.launches == 2 && execution.enqueues == 2 &&
      execution.commands_submitted == 2 * execution.commands_per_enqueue && !execution.fallback_count,
      "constant evaluation did not execute the expected native graph");
  std::ofstream output(argv[2], std::ios::binary);
  output.write(reference.data(), reference.size()); output.close();
  Require(bool(output), "failed to write computed constants");
  std::cout << "AGINFER_CONSTANT_EVAL_V1 " << outputs << ' ' << execution.commands_per_enqueue << " 2\n";
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
