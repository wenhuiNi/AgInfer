#include "aginfer/c_api.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int fail(ai_status status, const char* operation) {
  fprintf(stderr, "%s: %s: %s\n", operation, ai_status_name(status),
          ai_last_error());
  return 1;
}

static void configure_runtime(ai_runtime_options* options,
                              ai_provider_abi providers[2],
                              uint32_t arch) {
  ai_provider_abi_init(&providers[0]);
  providers[0].provider_id = AI_PROVIDER_CUBLASLT;
  providers[0].abi_version = 12;
  ai_provider_abi_init(&providers[1]);
  providers[1].provider_id = AI_PROVIDER_CUDNN;
  providers[1].abi_version = 9;
  ai_runtime_options_init(options);
  options->cuda_arch_override = arch;
  options->cuda_driver_version_override = 12000;
  options->cuda_runtime_version_override = 12000;
  options->provider_abis = providers;
  options->provider_abi_count = 2;
}

int main(int argc, char** argv) {
  ai_runtime_options short_options;
  ai_runtime_options_init(&short_options);
  short_options.struct_size = 8;
  ai_runtime* rejected_runtime = NULL;
  if (ai_runtime_create(&short_options, &rejected_runtime) !=
          AI_STATUS_INVALID_ARGUMENT ||
      rejected_runtime != NULL) {
    return 2;
  }
  ai_runtime_options_init(&short_options);
  short_options.struct_version = 99;
  if (ai_runtime_create(&short_options, &rejected_runtime) !=
          AI_STATUS_INCOMPATIBLE_ABI ||
      rejected_runtime != NULL) {
    return 3;
  }

  if (argc != 2) return 4;
  ai_provider_abi providers[2];
  ai_runtime_options options;
  configure_runtime(&options, providers, 89);
  ai_runtime* runtime = NULL;
  ai_status status = ai_runtime_create(&options, &runtime);
  if (status != AI_STATUS_OK) return 10 + fail(status, "runtime create");

  ai_model* model = NULL;
  status = ai_model_load(argv[1], &model);
  if (status != AI_STATUS_OK) return 20 + fail(status, "model load");

  ai_session_options session_options;
  ai_session_options_init(&session_options);
  ai_session* session = NULL;
  status = ai_session_create(runtime, model, &session_options, &session);
  if (status != AI_STATUS_OK) return 30 + fail(status, "session create");

  ai_target_info target;
  ai_target_info_init(&target);
  status = ai_session_get_target_info(session, &target);
  if (status != AI_STATUS_OK || target.cuda_arch != 89 ||
      target.platform != AI_PLATFORM_LINUX_X86_64_GNU ||
      target.runtime_abi != AI_RUNTIME_ABI_VERSION) {
    return 40 + fail(status, "target info");
  }

  ai_workspace_info workspace;
  ai_workspace_info_init(&workspace);
  status = ai_session_get_workspace_info(session, &workspace);
  if (status != AI_STATUS_OK || workspace.arena_bytes != 4096 ||
      workspace.workspace_bytes != 2048) {
    return 50 + fail(status, "workspace info");
  }

  uint32_t input_count = 0;
  uint32_t output_count = 0;
  if (ai_session_get_port_count(session, AI_PORT_INPUT, &input_count) !=
          AI_STATUS_OK ||
      ai_session_get_port_count(session, AI_PORT_OUTPUT, &output_count) !=
          AI_STATUS_OK ||
      input_count != 3 || output_count != 1) {
    return 60;
  }
  ai_port_info port;
  ai_port_info_init(&port);
  status = ai_session_get_port_info(session, AI_PORT_INPUT, 0, &port);
  if (status != AI_STATUS_OK || port.port_id != 0 ||
      port.dtype != AI_DTYPE_I32 || port.rank != 1 || port.shape[0] != 1 ||
      port.byte_size != 4 || strcmp(port.diagnostic_name, "input_ids") != 0) {
    return 61 + fail(status, "input port info");
  }

  ai_target_info short_target;
  ai_target_info_init(&short_target);
  short_target.struct_size = 8;
  if (ai_session_get_target_info(session, &short_target) !=
          AI_STATUS_INVALID_ARGUMENT ||
      ai_session_get_port_count(session, 99, &input_count) !=
          AI_STATUS_INVALID_ARGUMENT ||
      ai_session_enqueue(session, NULL) != AI_STATUS_INVALID_STATE) {
    return 70;
  }

  int64_t shape[1] = {1};
  int64_t stride[1] = {1};
  int value = 0;
  ai_tensor_view bad_view;
  ai_tensor_view_init(&bad_view);
  bad_view.dtype = 999;
  bad_view.location = AI_MEMORY_DEVICE;
  bad_view.rank = 1;
  bad_view.shape = shape;
  bad_view.stride = stride;
  bad_view.data = &value;
  bad_view.byte_size = sizeof(value);
  if (ai_session_bind_input(session, 0, &bad_view) !=
      AI_STATUS_INVALID_ARGUMENT) {
    return 71;
  }

  ai_session_destroy(session);
  ai_model_destroy(model);
  ai_runtime_destroy(runtime);

  configure_runtime(&options, providers, 89);
  providers[0].abi_version = 11;
  runtime = NULL;
  if (ai_runtime_create(&options, &runtime) != AI_STATUS_OK) return 80;
  if (ai_model_load(argv[1], &model) != AI_STATUS_OK) return 81;
  session = NULL;
  status = ai_session_create(runtime, model, NULL, &session);
  if (status != AI_STATUS_INCOMPATIBLE_ABI || session != NULL) return 82;
  ai_model_destroy(model);
  ai_runtime_destroy(runtime);

  configure_runtime(&options, providers, 110);
  runtime = NULL;
  if (ai_runtime_create(&options, &runtime) != AI_STATUS_OK) return 90;
  if (ai_model_load(argv[1], &model) != AI_STATUS_OK) return 91;
  session = NULL;
  status = ai_session_create(runtime, model, NULL, &session);
  if (status != AI_STATUS_INCOMPATIBLE_ARCHITECTURE || session != NULL) {
    return 92;
  }
  ai_model_destroy(model);
  ai_runtime_destroy(runtime);
  return 0;
}
