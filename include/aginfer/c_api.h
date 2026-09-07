#ifndef AGINFER_C_API_H_
#define AGINFER_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI_RUNTIME_ABI_VERSION 1u
#define AI_STRUCT_VERSION_1 1u
#define AI_DEFAULT_PROFILE UINT32_MAX
#define AI_MAX_TENSOR_RANK 8u

typedef uint32_t ai_status;
enum {
  AI_STATUS_OK = 0,
  AI_STATUS_INVALID_ARGUMENT = 1,
  AI_STATUS_NOT_FOUND = 2,
  AI_STATUS_IO_ERROR = 3,
  AI_STATUS_CORRUPT_PACKAGE = 4,
  AI_STATUS_INCOMPATIBLE_PLATFORM = 5,
  AI_STATUS_INCOMPATIBLE_ARCHITECTURE = 6,
  AI_STATUS_INCOMPATIBLE_ABI = 7,
  AI_STATUS_CUDA_ERROR = 8,
  AI_STATUS_OUT_OF_MEMORY = 9,
  AI_STATUS_INVALID_STATE = 10,
};

typedef uint32_t ai_dtype;
enum {
  AI_DTYPE_F32 = 1,
  AI_DTYPE_F16 = 2,
  AI_DTYPE_BF16 = 3,
  AI_DTYPE_F8_E4M3 = 4,
  AI_DTYPE_I32 = 5,
  AI_DTYPE_BOOL = 6,
};

typedef uint32_t ai_memory_location;
enum {
  AI_MEMORY_HOST = 1,
  AI_MEMORY_DEVICE = 2,
};

typedef uint32_t ai_port_kind;
enum {
  AI_PORT_INPUT = 1,
  AI_PORT_OUTPUT = 2,
};

typedef uint32_t ai_platform;
enum {
  AI_PLATFORM_LINUX_X86_64_GNU = 1,
  AI_PLATFORM_LINUX_AARCH64_SBSA = 2,
};

typedef uint32_t ai_provider_id;
enum {
  AI_PROVIDER_CUBLASLT = 1,
  AI_PROVIDER_CUDNN = 2,
};

typedef struct ai_runtime ai_runtime;
typedef struct ai_model ai_model;
typedef struct ai_session ai_session;

typedef struct ai_provider_abi {
  uint32_t struct_size;
  uint32_t struct_version;
  ai_provider_id provider_id;
  uint32_t abi_version;
} ai_provider_abi;

typedef struct ai_runtime_options {
  uint32_t struct_size;
  uint32_t struct_version;
  uint32_t required_runtime_abi;
  uint32_t flags;
  uint32_t cuda_arch_override;
  uint32_t cuda_driver_version_override;
  uint32_t cuda_runtime_version_override;
  uint32_t provider_abi_count;
  const ai_provider_abi* provider_abis;
} ai_runtime_options;

typedef struct ai_session_options {
  uint32_t struct_size;
  uint32_t struct_version;
  uint32_t profile_index;
  uint32_t flags;  // Reserved; must be zero. V2 sessions always use CUDA Graph.
} ai_session_options;

typedef struct ai_tensor_view {
  uint32_t struct_size;
  uint32_t struct_version;
  ai_dtype dtype;
  ai_memory_location location;
  uint32_t rank;
  uint32_t flags;
  const int64_t* shape;
  const int64_t* stride;
  void* data;
  uint64_t byte_size;
} ai_tensor_view;

typedef struct ai_port_info {
  uint32_t struct_size;
  uint32_t struct_version;
  uint32_t port_id;
  ai_port_kind kind;
  ai_dtype dtype;
  ai_memory_location location;
  uint32_t rank;
  uint32_t flags;
  uint64_t byte_size;
  int64_t shape[AI_MAX_TENSOR_RANK];
  int64_t stride[AI_MAX_TENSOR_RANK];
  const char* diagnostic_name;
} ai_port_info;

typedef struct ai_target_info {
  uint32_t struct_size;
  uint32_t struct_version;
  ai_platform platform;
  uint32_t cuda_arch;
  uint32_t runtime_abi;
  uint32_t flags;
} ai_target_info;

typedef struct ai_workspace_info {
  uint32_t struct_size;
  uint32_t struct_version;
  uint64_t arena_bytes;
  uint64_t workspace_bytes;
} ai_workspace_info;

// Logical commands submitted directly or through successful internal graph
// launches, not GPU completion counts. Capture/Prepare does not increment them.
// A v2 session has no fallback implementation. Query only when not enqueueing.
typedef struct ai_execution_info {
  uint32_t struct_size;
  uint32_t struct_version;
  uint64_t commands_per_enqueue;
  uint64_t enqueues;
  uint64_t commands_submitted;
  uint64_t fallback_count;
} ai_execution_info;

void ai_execution_info_init(ai_execution_info* value);
// provider_id=0 returns the entire plan; otherwise use its numeric command provider ID.
ai_status ai_session_get_execution_info(const ai_session* session, uint32_t provider_id,
                                       ai_execution_info* info);

typedef struct ai_cuda_graph_info {
  uint32_t struct_size;
  uint32_t struct_version;
  uint32_t enabled;
  uint32_t instantiated;
  uint64_t node_count;
  uint64_t launches;
} ai_cuda_graph_info;

void ai_cuda_graph_info_init(ai_cuda_graph_info* value);
// V2 only. Successful launch submissions are not proof of GPU completion.
ai_status ai_session_get_cuda_graph_info(const ai_session* session, ai_cuda_graph_info* info);

void ai_provider_abi_init(ai_provider_abi* value);
void ai_runtime_options_init(ai_runtime_options* value);
void ai_session_options_init(ai_session_options* value);
void ai_tensor_view_init(ai_tensor_view* value);
void ai_port_info_init(ai_port_info* value);
void ai_target_info_init(ai_target_info* value);
void ai_workspace_info_init(ai_workspace_info* value);

ai_status ai_runtime_create(const ai_runtime_options* options,
                            ai_runtime** runtime_out);
void ai_runtime_destroy(ai_runtime* runtime);

ai_status ai_model_load(const char* aim_path, ai_model** model_out);
void ai_model_destroy(ai_model* model);

ai_status ai_session_create(ai_runtime* runtime, ai_model* model,
                            const ai_session_options* options,
                            ai_session** session_out);
void ai_session_destroy(ai_session* session);
ai_status ai_session_prepare(ai_session* session);
ai_status ai_session_get_target_info(const ai_session* session,
                                     ai_target_info* info);
ai_status ai_session_get_workspace_info(const ai_session* session,
                                        ai_workspace_info* info);
ai_status ai_session_get_port_count(const ai_session* session,
                                    ai_port_kind kind, uint32_t* count);
ai_status ai_session_get_port_info(const ai_session* session,
                                   ai_port_kind kind, uint32_t index,
                                   ai_port_info* info);
ai_status ai_session_bind_input(ai_session* session, uint32_t port_id,
                                const ai_tensor_view* view);
ai_status ai_session_bind_output(ai_session* session, uint32_t port_id,
                                 const ai_tensor_view* view);
ai_status ai_session_enqueue(ai_session* session, void* cuda_stream);
const char* ai_session_last_error(const ai_session* session);

const char* ai_last_error(void);
const char* ai_status_name(ai_status status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AGINFER_C_API_H_
