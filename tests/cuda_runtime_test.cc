#include "aginfer/runtime.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool CudaOk(cudaError_t result, const char* operation) {
  if (result == cudaSuccess) return true;
  std::cerr << operation << ": " << cudaGetErrorString(result) << '\n';
  return false;
}

ai_tensor_view View(float* data, const std::int64_t* shape,
                    const std::int64_t* stride) {
  ai_tensor_view view;
  ai_tensor_view_init(&view);
  view.dtype = AI_DTYPE_F32;
  view.location = AI_MEMORY_DEVICE;
  view.rank = 1;
  view.shape = shape;
  view.stride = stride;
  view.data = data;
  view.byte_size = 16 * sizeof(float);
  return view;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  ai_runtime_options options;
  ai_runtime_options_init(&options);
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
  status = session.Prepare();
  if (status != AI_STATUS_OK) {
    std::cerr << session.last_error() << '\n';
    return 6;
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
  if (!CudaOk(cudaMalloc(reinterpret_cast<void**>(&device_left), bytes),
              "cudaMalloc(left)") ||
      !CudaOk(cudaMalloc(reinterpret_cast<void**>(&device_right), bytes),
              "cudaMalloc(right)") ||
      !CudaOk(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes),
              "cudaMalloc(output)")) {
    return 7;
  }
  if (!CudaOk(cudaMemcpy(device_left, left.data(), bytes,
                         cudaMemcpyHostToDevice),
              "copy left") ||
      !CudaOk(cudaMemcpy(device_right, right.data(), bytes,
                         cudaMemcpyHostToDevice),
              "copy right")) {
    return 8;
  }

  const std::int64_t shape[1] = {16};
  const std::int64_t stride[1] = {1};
  const ai_tensor_view left_view = View(device_left, shape, stride);
  const ai_tensor_view right_view = View(device_right, shape, stride);
  const ai_tensor_view output_view = View(device_output, shape, stride);
  if (session.BindInput(0, left_view) != AI_STATUS_OK ||
      session.BindInput(1, right_view) != AI_STATUS_OK ||
      session.BindOutput(2, output_view) != AI_STATUS_OK) {
    return 9;
  }
  status = session.Enqueue();
  if (status != AI_STATUS_OK) {
    std::cerr << session.last_error() << '\n';
    return 10;
  }
  if (!CudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
      !CudaOk(cudaMemcpy(actual.data(), device_output, bytes,
                         cudaMemcpyDeviceToHost),
              "copy output")) {
    return 11;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (std::fabs(actual[index] - expected[index]) > 1e-6F) return 12;
  }

  const std::int64_t wrong_shape[1] = {15};
  const ai_tensor_view bad_view = View(device_left, wrong_shape, stride);
  if (session.BindInput(0, bad_view) != AI_STATUS_INVALID_ARGUMENT) return 13;
  cudaFree(device_output);
  cudaFree(device_right);
  cudaFree(device_left);
  return 0;
}
