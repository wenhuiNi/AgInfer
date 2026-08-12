#include "aginfer/runtime.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool CudaOk(cudaError_t result, const char* operation) {
  if (result == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(result) << '\n';
  return false;
}

aginfer::TensorView View(const char* name, float* data) {
  return {name, aginfer::DType::kFp32, {16}, {1}, data,
          16 * sizeof(float), aginfer::MemoryLocation::kDevice, true};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  aginfer::RuntimeOptions options;
  // The execution path itself uses only the CUDA Driver. These explicit ABI
  // values keep the test independent of optional cuDNN installation.
  options.cublaslt_abi_override = 12;
  options.cudnn_abi_override = 9;
  auto runtime = aginfer::Runtime::Create(options);
  if (!runtime.ok()) {
    std::cerr << runtime.status().message() << '\n';
    return 3;
  }
  auto model = aginfer::Model::Load(argv[1]);
  if (!model.ok()) {
    std::cerr << model.status().message() << '\n';
    return 4;
  }
  aginfer::SessionOptions session_options;
  session_options.profile = "vector16";
  auto session = aginfer::Session::Create(runtime.value(), model.value(), session_options);
  if (!session.ok()) {
    std::cerr << session.status().message() << '\n';
    return 5;
  }

  std::vector<float> left(16), right(16), expected(16), actual(16);
  for (std::size_t index = 0; index < left.size(); ++index) {
    left[index] = static_cast<float>(index) * 0.5F;
    right[index] = static_cast<float>(index) * -0.25F;
    expected[index] = left[index] + right[index];
  }
  float* device_left = nullptr;
  float* device_right = nullptr;
  float* device_output = nullptr;
  const std::size_t bytes = left.size() * sizeof(float);
  if (!CudaOk(cudaMalloc(reinterpret_cast<void**>(&device_left), bytes), "cudaMalloc(left)") ||
      !CudaOk(cudaMalloc(reinterpret_cast<void**>(&device_right), bytes), "cudaMalloc(right)") ||
      !CudaOk(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes), "cudaMalloc(output)")) return 6;
  if (!CudaOk(cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice), "copy left") ||
      !CudaOk(cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice), "copy right")) return 7;

  std::vector<aginfer::TensorView> inputs{View("left", device_left), View("right", device_right)};
  std::vector<aginfer::TensorView> outputs{View("output", device_output)};
  aginfer::RunOptions run_options;
  const aginfer::Status enqueue = session.value().Enqueue(inputs, outputs, run_options, nullptr);
  if (!enqueue.ok()) {
    std::cerr << enqueue.message() << '\n';
    return 8;
  }
  if (!CudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
      !CudaOk(cudaMemcpy(actual.data(), device_output, bytes, cudaMemcpyDeviceToHost), "copy output")) return 9;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::fabs(actual[index] - expected[index]) > 1e-6F) return 10;
  }

  inputs[0].shape[0] = 15;
  const aginfer::Status bad_shape = session.value().Enqueue(inputs, outputs, run_options, nullptr);
  if (bad_shape.ok() || bad_shape.code() != aginfer::StatusCode::kInvalidArgument) return 11;
  cudaFree(device_output);
  cudaFree(device_right);
  cudaFree(device_left);
  return 0;
}
