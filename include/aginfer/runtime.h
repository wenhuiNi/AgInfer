#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aginfer {

inline constexpr std::uint32_t kRuntimeAbi = 1;

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kIoError,
  kCorruptPackage,
  kIncompatiblePlatform,
  kIncompatibleArchitecture,
  kIncompatibleAbi,
  kCudaError,
  kOutOfMemory,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return {}; }
  bool ok() const noexcept { return code_ == StatusCode::kOk; }
  StatusCode code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

template <typename T>
class StatusOr {
 public:
  StatusOr(Status status) : status_(std::move(status)) {}
  StatusOr(T value) : status_(), value_(std::make_unique<T>(std::move(value))) {}

  bool ok() const noexcept { return status_.ok(); }
  const Status& status() const noexcept { return status_; }
  T& value() & { return *value_; }
  const T& value() const& { return *value_; }
  T&& value() && { return std::move(*value_); }

 private:
  Status status_;
  std::unique_ptr<T> value_;
};

enum class DType { kFp32, kFp16, kBf16, kFp8E4M3, kInt32 };
enum class MemoryLocation { kHost, kDevice };
enum class MathMode { kStrict, kTf32 };

struct TensorView {
  std::string name;
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride;
  void* data = nullptr;
  std::size_t byte_size = 0;
  MemoryLocation location = MemoryLocation::kDevice;
  bool caller_owned = true;
};

struct TensorInfo {
  std::string name;
  DType dtype = DType::kFp32;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride;
  std::size_t byte_size = 0;
  MemoryLocation location = MemoryLocation::kDevice;
};

struct RunOptions {
  std::uint32_t denoising_steps = 0;  // Zero selects the manifest default.
  std::uint64_t seed = 0;
  bool deterministic = true;
  MathMode math_mode = MathMode::kStrict;
  const TensorView* initial_noise_override = nullptr;
};

struct RuntimeOptions {
  std::uint32_t required_runtime_abi = kRuntimeAbi;
  // Non-zero values override probes for tests/embedded launchers. Normal
  // applications leave them at zero and let the runtime inspect the system.
  std::uint32_t cuda_arch_override = 0;
  std::uint32_t cuda_driver_version_override = 0;
  std::uint32_t cuda_runtime_version_override = 0;
  std::uint32_t cublaslt_abi_override = 0;
  std::uint32_t cudnn_abi_override = 0;
};

struct SessionOptions {
  std::string profile;
};

struct TargetInfo {
  std::string platform;
  std::string cuda_arch;
  std::uint32_t runtime_abi = 0;
};

struct WorkspaceInfo {
  std::uint64_t arena_bytes = 0;
  std::uint64_t workspace_bytes = 0;
};

using CudaStream = void*;

class Runtime {
 public:
  Runtime(Runtime&&) noexcept;
  Runtime& operator=(Runtime&&) noexcept;
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  static StatusOr<Runtime> Create(const RuntimeOptions& options = {});

 private:
  struct Impl;
  explicit Runtime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Session;
};

class Model {
 public:
  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  static StatusOr<Model> Load(const std::string& aim_path);

 private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Session;
};

class Session {
 public:
  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  static StatusOr<Session> Create(Runtime& runtime, Model& model,
                                  const SessionOptions& options = {});
  TargetInfo GetTargetInfo() const;
  std::vector<TensorInfo> GetInputInfo() const;
  std::vector<TensorInfo> GetOutputInfo() const;
  StatusOr<WorkspaceInfo> GetRequiredWorkspace(const std::string& profile) const;
  Status Enqueue(const std::vector<TensorView>& inputs,
                 const std::vector<TensorView>& outputs,
                 const RunOptions& options, CudaStream stream);

 private:
  struct Impl;
  explicit Session(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

const char* StatusCodeName(StatusCode code) noexcept;

}  // namespace aginfer
