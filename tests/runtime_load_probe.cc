#include "aginfer/c_api.h"

#include <iostream>

// CPU-only load probe: no runtime/session creation and no GPU access.
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  ai_model* model = nullptr;
  const ai_status status = ai_model_load(argv[1], &model);
  std::cout << ai_status_name(status) << '\n' << ai_last_error() << '\n';
  if ((status == AI_STATUS_OK) != (model != nullptr)) return 3;
  ai_model_destroy(model);
  return 0;
}
