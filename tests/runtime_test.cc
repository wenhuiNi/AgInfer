#include "aginfer/runtime.hpp"

#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  ai_provider_abi providers[2];
  ai_provider_abi_init(&providers[0]);
  providers[0].provider_id = AI_PROVIDER_CUBLASLT;
  providers[0].abi_version = 12;
  ai_provider_abi_init(&providers[1]);
  providers[1].provider_id = AI_PROVIDER_CUDNN;
  providers[1].abi_version = 9;

  ai_runtime_options options;
  ai_runtime_options_init(&options);
  options.cuda_arch_override = 89;
  options.cuda_driver_version_override = 12000;
  options.cuda_runtime_version_override = 12000;
  options.provider_abis = providers;
  options.provider_abi_count = 2;

  aginfer::Runtime runtime;
  ai_status status = aginfer::Runtime::Create(&options, &runtime);
  if (status != AI_STATUS_OK) {
    std::cerr << ai_status_name(status) << ": " << ai_last_error() << '\n';
    return 3;
  }
  aginfer::Model model;
  status = aginfer::Model::Load(argv[1], &model);
  if (status != AI_STATUS_OK) return 4;
  aginfer::Session session;
  status = aginfer::Session::Create(runtime, model, nullptr, &session);
  if (status != AI_STATUS_OK) return 5;
  ai_session_options graph_options; ai_session_options_init(&graph_options);
  graph_options.flags = AI_SESSION_CUDA_GRAPH;
  ai_session* rejected = nullptr;
  if (ai_session_create(runtime.get(), model.get(), &graph_options, &rejected) != AI_STATUS_INVALID_ARGUMENT || rejected)
    return 7;

  ai_port_info output;
  ai_port_info_init(&output);
  status = ai_session_get_port_info(session.get(), AI_PORT_OUTPUT, 0, &output);
  if (status != AI_STATUS_OK || output.port_id != 3 ||
      output.dtype != AI_DTYPE_F16 || output.byte_size != 2 ||
      std::strcmp(output.diagnostic_name, "actions") != 0) {
    return 6;
  }
  return 0;
}
