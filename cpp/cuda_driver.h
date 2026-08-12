#pragma once

#include "aginfer/runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace aginfer::internal {

using CuDevicePtr = std::uint64_t;
using CuModule = void*;
using CuFunction = void*;

class CudaDriver {
 public:
  CudaDriver(CudaDriver&&) noexcept;
  CudaDriver& operator=(CudaDriver&&) noexcept;
  ~CudaDriver();
  CudaDriver(const CudaDriver&) = delete;
  CudaDriver& operator=(const CudaDriver&) = delete;

  static StatusOr<CudaDriver> Create(std::uint32_t expected_arch);
  StatusOr<CuModule> LoadModule(const void* image);
  StatusOr<CuFunction> GetFunction(CuModule module, const std::string& name);
  Status UnloadModule(CuModule module);
  StatusOr<CuDevicePtr> Allocate(std::uint64_t bytes);
  Status Free(CuDevicePtr pointer);
  Status CopyHostToDevice(CuDevicePtr destination, const void* source, std::size_t bytes);
  Status Launch(CuFunction function, const std::uint32_t grid[3],
                const std::uint32_t block[3], std::uint32_t shared_bytes,
                CudaStream stream, void** arguments);
  Status MakeCurrent();

 private:
  struct Impl;
  explicit CudaDriver(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace aginfer::internal

