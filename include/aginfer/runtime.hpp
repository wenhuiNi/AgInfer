#pragma once

#include "aginfer/c_api.h"

#include <utility>

namespace aginfer {

class Runtime {
 public:
  Runtime() = default;
  Runtime(Runtime&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Runtime& operator=(Runtime&& other) noexcept {
    if (this != &other) {
      ai_runtime_destroy(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  ~Runtime() { ai_runtime_destroy(handle_); }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  static ai_status Create(const ai_runtime_options* options, Runtime* output) {
    if (output == nullptr) return AI_STATUS_INVALID_ARGUMENT;
    ai_runtime* handle = nullptr;
    const ai_status status = ai_runtime_create(options, &handle);
    if (status == AI_STATUS_OK) {
      ai_runtime_destroy(output->handle_);
      output->handle_ = handle;
    }
    return status;
  }
  ai_runtime* get() const noexcept { return handle_; }

 private:
  ai_runtime* handle_ = nullptr;
};

class Model {
 public:
  Model() = default;
  Model(Model&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Model& operator=(Model&& other) noexcept {
    if (this != &other) {
      ai_model_destroy(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  ~Model() { ai_model_destroy(handle_); }
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  static ai_status Load(const char* path, Model* output) {
    if (output == nullptr) return AI_STATUS_INVALID_ARGUMENT;
    ai_model* handle = nullptr;
    const ai_status status = ai_model_load(path, &handle);
    if (status == AI_STATUS_OK) {
      ai_model_destroy(output->handle_);
      output->handle_ = handle;
    }
    return status;
  }
  ai_model* get() const noexcept { return handle_; }

 private:
  ai_model* handle_ = nullptr;
};

class Session {
 public:
  Session() = default;
  Session(Session&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Session& operator=(Session&& other) noexcept {
    if (this != &other) {
      ai_session_destroy(handle_);
      handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
  }
  ~Session() { ai_session_destroy(handle_); }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  static ai_status Create(Runtime& runtime, Model& model,
                          const ai_session_options* options,
                          Session* output) {
    if (output == nullptr) return AI_STATUS_INVALID_ARGUMENT;
    ai_session* handle = nullptr;
    const ai_status status =
        ai_session_create(runtime.get(), model.get(), options, &handle);
    if (status == AI_STATUS_OK) {
      ai_session_destroy(output->handle_);
      output->handle_ = handle;
    }
    return status;
  }

  ai_status Prepare() { return ai_session_prepare(handle_); }
  ai_status BindInput(uint32_t id, const ai_tensor_view& view) {
    return ai_session_bind_input(handle_, id, &view);
  }
  ai_status BindOutput(uint32_t id, const ai_tensor_view& view) {
    return ai_session_bind_output(handle_, id, &view);
  }
  ai_status Enqueue(void* stream = nullptr) {
    return ai_session_enqueue(handle_, stream);
  }
  const char* last_error() const { return ai_session_last_error(handle_); }
  ai_session* get() const noexcept { return handle_; }

 private:
  ai_session* handle_ = nullptr;
};

}  // namespace aginfer
