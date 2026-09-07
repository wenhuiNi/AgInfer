#include "aginfer/c_api.h"
#include <cstdint>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  ai_runtime_options options; ai_runtime_options_init(&options);
  options.cuda_arch_override = 120; options.cuda_driver_version_override = 13000;
  options.cuda_runtime_version_override = 12080;
  ai_runtime* runtime = nullptr; ai_model* model = nullptr; ai_session* session = nullptr;
  if (ai_runtime_create(&options, &runtime) || ai_model_load(argv[1], &model) ||
      ai_session_create(runtime, model, nullptr, &session)) return 3;
  std::uint32_t count = 0;
  if (ai_session_get_port_count(session, AI_PORT_INPUT, &count) || count != 1) return 4;
  ai_port_info info; ai_port_info_init(&info);
  if (ai_session_get_port_info(session, AI_PORT_INPUT, 0, &info) ||
      info.port_id != 0 || info.dtype != AI_DTYPE_F32 || info.rank != 1 ||
      info.shape[0] != 1 || info.stride[0] != 1 || info.byte_size != 4) return 5;
  if (ai_session_enqueue(session, nullptr) != AI_STATUS_INVALID_STATE ||
      ai_session_prepare(session) != AI_STATUS_INVALID_STATE) return 6;
  ai_tensor_view view; ai_tensor_view_init(&view);
  view.dtype = info.dtype; view.location = info.location; view.rank = info.rank;
  view.shape = info.shape; view.stride = info.stride; view.byte_size = info.byte_size;
  // No GPU launch or allocation in this public CPU contract test.
  view.data = reinterpret_cast<void*>(std::uintptr_t{0x1000});
  if (ai_session_bind_input(session, 0, &view)) return 7;
  view.dtype = AI_DTYPE_BOOL;
  if (ai_session_bind_input(session, 0, &view) != AI_STATUS_INVALID_ARGUMENT) return 8;
  if (ai_session_prepare(session) != AI_STATUS_INVALID_STATE) return 9;
  ai_execution_info execution; ai_execution_info_init(&execution);
  if (ai_session_get_execution_info(session, 0, &execution) || execution.commands_per_enqueue != 2 ||
      execution.enqueues || execution.commands_submitted || execution.fallback_count) return 10;
  if (ai_session_get_execution_info(session, 99, &execution) != AI_STATUS_INVALID_ARGUMENT) return 11;
  ai_session_destroy(session); ai_model_destroy(model); ai_runtime_destroy(runtime);
  return 0;
}
