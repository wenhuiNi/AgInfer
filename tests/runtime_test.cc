#include "aginfer/runtime.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  auto model = aginfer::Model::Load(argv[1]);
  if (!model.ok()) {
    std::cerr << aginfer::StatusCodeName(model.status().code()) << ": " << model.status().message() << '\n';
    return 3;
  }
  aginfer::RuntimeOptions options;
  options.cuda_arch_override = 89;
  options.cuda_driver_version_override = 12000;
  options.cuda_runtime_version_override = 12000;
  options.cublaslt_abi_override = 12;
  options.cudnn_abi_override = 9;
  auto runtime = aginfer::Runtime::Create(options);
  if (!runtime.ok()) return 4;
  auto session = aginfer::Session::Create(runtime.value(), model.value());
  if (!session.ok()) {
    std::cerr << aginfer::StatusCodeName(session.status().code()) << ": " << session.status().message() << '\n';
    return 5;
  }
  if (session.value().GetTargetInfo().cuda_arch != "sm89") return 6;
  const auto inputs = session.value().GetInputInfo();
  const auto outputs = session.value().GetOutputInfo();
  if (inputs.size() != 3 || inputs[0].name != "input_ids" || inputs[0].dtype != aginfer::DType::kInt32 ||
      inputs[0].shape != std::vector<std::int64_t>({1}) || inputs[0].byte_size != 4) return 9;
  if (outputs.size() != 1 || outputs[0].name != "actions" || outputs[0].dtype != aginfer::DType::kFp16 ||
      outputs[0].shape != std::vector<std::int64_t>({1}) || outputs[0].byte_size != 2) return 10;
  auto workspace = session.value().GetRequiredWorkspace("default");
  if (!workspace.ok() || workspace.value().arena_bytes != 4096 || workspace.value().workspace_bytes != 2048) return 7;

  options.cuda_arch_override = 110;
  auto wrong_runtime = aginfer::Runtime::Create(options);
  auto wrong_session = aginfer::Session::Create(wrong_runtime.value(), model.value());
  if (wrong_session.ok() || wrong_session.status().code() != aginfer::StatusCode::kIncompatibleArchitecture) return 8;
  return 0;
}
