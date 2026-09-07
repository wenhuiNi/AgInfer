#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace aginfer::internal {

enum class StatusCode : std::uint32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kNotFound = 2,
  kIoError = 3,
  kCorruptPackage = 4,
  kIncompatiblePlatform = 5,
  kIncompatibleArchitecture = 6,
  kIncompatibleAbi = 7,
  kCudaError = 8,
  kOutOfMemory = 9,
  kInvalidState = 10,
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
  StatusOr(T value) : value_(std::make_unique<T>(std::move(value))) {}

  bool ok() const noexcept { return status_.ok(); }
  const Status& status() const noexcept { return status_; }
  T& value() & { return *value_; }
  const T& value() const& { return *value_; }
  T&& value() && { return std::move(*value_); }

 private:
  Status status_;
  std::unique_ptr<T> value_;
};

}  // namespace aginfer::internal
