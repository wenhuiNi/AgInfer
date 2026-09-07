cmake_minimum_required(VERSION 3.20)
project(AgInfer VERSION 0.1.0 LANGUAGES C CXX)

option(AGINFER_ENABLE_CUDA
       "Build the CUDA providers and target-architecture AOT kernels" ON)
option(AGINFER_ENABLE_FLASHINFER_ATTENTION
       "Build the pinned FlashInfer attention provider when targeting SM120" ON)
option(AGINFER_BUILD_COMPILER_TOOLS "Build offline CUDA compiler selection tools" ON)
option(AGINFER_VERIFY_AIM_CHECKSUMS
       "Verify all AIM container checksums on load (always enabled in Debug)" OFF)
set(AGINFER_AIM_CHECKSUMS_ENABLED
    "$<OR:$<CONFIG:Debug>,$<BOOL:${AGINFER_VERIFY_AIM_CHECKSUMS}>>")
set(AGINFER_CUDA_ARCH "120" CACHE STRING
    "CUDA architecture number for the deployed AOT kernel bundle")

add_library(aginfer_runtime STATIC
  cpp/action_slice_payload.cc
  cpp/adaptive_rms_norm_payload.cc
  cpp/command_stream.cc
  cpp/cublaslt_payload.cc
  cpp/cuda_driver.cc
  cpp/executable_plan.cc
  cpp/executable_session.cc
  cpp/flashinfer_attention_payload.cc
  cpp/kv_pack_payload.cc
  cpp/patch_projection_payload.cc
  cpp/projection_split_payload.cc
  cpp/state_update_payload.cc
  cpp/prefix_input_payload.cc
  cpp/prefix_kv_store_payload.cc
  cpp/layer_norm_payload.cc
  cpp/rms_norm_payload.cc
  cpp/rope_payload.cc
  cpp/rounded_attention_payload.cc
  cpp/suffix_metadata_payload.cc
  cpp/time_embedding_payload.cc
  cpp/vision_attention_payload.cc
  cpp/plan.cc
  cpp/providers/dispatch.cc
  cpp/runtime.cc
  cpp/sha256.cc)
target_compile_features(aginfer_runtime PRIVATE cxx_std_20)
target_compile_definitions(aginfer_runtime PRIVATE
  $<${AGINFER_AIM_CHECKSUMS_ENABLED}:AGINFER_VERIFY_AIM_CHECKSUMS=1>)
target_include_directories(aginfer_runtime PUBLIC include PRIVATE cpp)
target_compile_options(aginfer_runtime PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
target_link_libraries(aginfer_runtime PRIVATE ${CMAKE_DL_LIBS})
if(AGINFER_ENABLE_CUDA)
  target_compile_definitions(aginfer_runtime PRIVATE AGINFER_HAS_CUDA_PROVIDERS=1)
  find_package(CUDAToolkit REQUIRED)
  if(NOT AGINFER_CUDA_ARCH MATCHES "^(89|110|120)$")
    message(FATAL_ERROR "AGINFER_CUDA_ARCH must be one of 89, 110, or 120")
  endif()
  target_sources(aginfer_runtime PRIVATE
    cpp/cuda_kernel_payload.cc
    cpp/providers/aot_action_slice.cc
    cpp/providers/aot_adaptive_rms_norm.cc
    cpp/providers/aot_activation.cc
    cpp/providers/aot_cast.cc
    cpp/providers/aot_layer_norm.cc
    cpp/providers/aot_kv_pack.cc
    cpp/providers/patch_projection.cc
    cpp/providers/projection_split.cc
    cpp/providers/state_update.cc
    cpp/providers/aot_prefix_input.cc
    cpp/providers/aot_prefix_kv_store.cc
    cpp/providers/aot_pointwise.cc
    cpp/providers/aot_rms_norm.cc
    cpp/providers/aot_rope.cc
    cpp/providers/rounded_attention.cc
    cpp/providers/aot_suffix_metadata.cc
    cpp/providers/aot_time_embedding.cc
    cpp/providers/aot_vision_attention.cc
    cpp/providers/cublaslt_linear.cc)
  target_link_libraries(aginfer_runtime PRIVATE CUDA::cublasLt CUDA::cudart)
  if(AGINFER_BUILD_COMPILER_TOOLS)
    add_executable(aginfer_cublaslt_select tools/cublaslt_select.cc)
    target_compile_features(aginfer_cublaslt_select PRIVATE cxx_std_17)
    target_compile_options(aginfer_cublaslt_select PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(aginfer_cublaslt_select PRIVATE CUDA::cublasLt CUDA::cudart)
  endif()

  if(AGINFER_ENABLE_FLASHINFER_ATTENTION)
    if(NOT AGINFER_CUDA_ARCH STREQUAL "120")
      message(STATUS
              "FlashInfer attention is currently delivered only for exact SM120")
    else()
      set(AGINFER_FLASHINFER_ROOT
          ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashinfer)
      set(AGINFER_CCCL_ROOT
          ${AGINFER_FLASHINFER_ROOT}/3rdparty/cccl)
      if(NOT EXISTS ${AGINFER_FLASHINFER_ROOT}/include/flashinfer/attention/prefill.cuh
         OR NOT EXISTS ${AGINFER_CCCL_ROOT}/libcudacxx/include/cuda/cmath)
        message(FATAL_ERROR
          "FlashInfer attention requires the pinned submodules; run: "
          "git submodule update --init third_party/flashinfer && "
          "git -C third_party/flashinfer submodule update --init 3rdparty/cccl")
      endif()
      enable_language(CUDA)
      target_compile_definitions(aginfer_runtime PRIVATE AGINFER_HAS_FLASHINFER=1)
      target_sources(aginfer_runtime PRIVATE
        cpp/providers/flashinfer_attention.cu)
      target_include_directories(aginfer_runtime PRIVATE
        ${AGINFER_CCCL_ROOT}/libcudacxx/include
        ${AGINFER_CCCL_ROOT}/cub
        ${AGINFER_CCCL_ROOT}/thrust
        ${AGINFER_FLASHINFER_ROOT}/include)
      # SM120a is an architecture-accelerated target and must be real SASS only.
      # Disable CMake's virtual-code pair and provide the exact nvcc target.
      set_property(TARGET aginfer_runtime PROPERTY CUDA_ARCHITECTURES OFF)
      set_property(TARGET aginfer_runtime PROPERTY CUDA_STANDARD 17)
      set_property(TARGET aginfer_runtime PROPERTY CUDA_STANDARD_REQUIRED ON)
      target_compile_options(aginfer_runtime PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_120a,code=sm_120a>)
    endif()
  endif()

  set(AGINFER_AOT_CUBIN
      ${CMAKE_CURRENT_BINARY_DIR}/kernels/aginfer-aot-sm${AGINFER_CUDA_ARCH}.cubin)
  add_custom_command(
    OUTPUT ${AGINFER_AOT_CUBIN}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/kernels
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -cubin
            -arch=sm_${AGINFER_CUDA_ARCH}
            --std=c++17
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/aot_kernels.cu
            -o ${AGINFER_AOT_CUBIN}
    DEPENDS kernels/aot_kernels.cu kernels/action_slice.cu kernels/activation.cu kernels/adaptive_rms_norm.cu
            kernels/cast.cu kernels/layer_norm.cu
            kernels/kv_pack.cu kernels/patch_projection.cu kernels/pointwise.cu kernels/prefix_input.cu
            kernels/prefix_kv_store.cu kernels/projection_split.cu kernels/state_update.cu
            kernels/rms_norm.cu kernels/rope.cu kernels/rounded_attention.cu
            kernels/suffix_metadata.cu
            kernels/time_embedding.cu
            kernels/vision_attention.cu
    VERBATIM)
  add_custom_target(aginfer_aot_kernels ALL DEPENDS ${AGINFER_AOT_CUBIN})
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(AGINFER_KERNEL_BUILD_RECORD "${AGINFER_AOT_CUBIN}.build.json")
  add_custom_command(
    OUTPUT ${AGINFER_KERNEL_BUILD_RECORD}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} -m aginfer.compiler.kernel_build
            --root ${CMAKE_CURRENT_SOURCE_DIR} --cubin ${AGINFER_AOT_CUBIN}
            --nvcc ${CUDAToolkit_NVCC_EXECUTABLE} --arch ${AGINFER_CUDA_ARCH}
            --output ${AGINFER_KERNEL_BUILD_RECORD}
    DEPENDS ${AGINFER_AOT_CUBIN} CMakeLists.txt
            src/aginfer/compiler/kernel_build.py src/aginfer/compiler/identity.py
    VERBATIM)
  add_custom_target(aginfer_kernel_build_record ALL DEPENDS ${AGINFER_KERNEL_BUILD_RECORD})
  # Both targets consume the CUBIN: order them so Make cannot launch its
  # generation rule twice in parallel through separate target makefiles.
  add_dependencies(aginfer_kernel_build_record aginfer_aot_kernels)
endif()

include(CTest)
option(AGINFER_BUILD_CUDA_TESTS "Build the real target-arch CUDA execution test" OFF)
set(AGINFER_CUDA_TEST_ARCH "${AGINFER_CUDA_ARCH}" CACHE STRING
    "CUDA architecture number for the synthetic execution test")
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/projection-split.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_projection_split_fixture.py
      ${CMAKE_CURRENT_BINARY_DIR}/projection-split.bin
    DEPENDS tests/make_projection_split_fixture.py src/aginfer/providers/projection_split.py VERBATIM)
  add_custom_target(projection_split_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/projection-split.bin)
  add_executable(projection_split_test tests/projection_split_test.cc)
  target_compile_features(projection_split_test PRIVATE cxx_std_20)
  target_include_directories(projection_split_test PRIVATE cpp)
  target_link_libraries(projection_split_test PRIVATE aginfer_runtime)
  add_dependencies(projection_split_test projection_split_fixture)
  add_test(NAME projection_split_test COMMAND projection_split_test ${CMAKE_CURRENT_BINARY_DIR}/projection-split.bin)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/state-update.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_state_update_fixture.py
      ${CMAKE_CURRENT_BINARY_DIR}/state-update.bin
    DEPENDS tests/make_state_update_fixture.py src/aginfer/providers/state_update.py VERBATIM)
  add_custom_target(state_update_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/state-update.bin)
  add_executable(state_update_test tests/state_update_test.cc)
  target_compile_features(state_update_test PRIVATE cxx_std_20)
  target_include_directories(state_update_test PRIVATE cpp)
  target_link_libraries(state_update_test PRIVATE aginfer_runtime)
  add_dependencies(state_update_test state_update_fixture)
  add_test(NAME state_update_test COMMAND state_update_test ${CMAKE_CURRENT_BINARY_DIR}/state-update.bin)
  if(AGINFER_ENABLE_CUDA AND AGINFER_BUILD_CUDA_TESTS AND AGINFER_CUDA_ARCH STREQUAL "120")
    add_executable(state_update_cuda_test tests/state_update_test.cc)
    target_compile_definitions(state_update_cuda_test PRIVATE TEST_CUDA=1)
    target_compile_features(state_update_cuda_test PRIVATE cxx_std_20)
    target_include_directories(state_update_cuda_test PRIVATE cpp)
    target_link_libraries(state_update_cuda_test PRIVATE aginfer_runtime CUDA::cudart)
    add_dependencies(state_update_cuda_test state_update_fixture aginfer_aot_kernels)
    add_test(NAME state_update_cuda_test COMMAND state_update_cuda_test
      ${CMAKE_CURRENT_BINARY_DIR}/state-update.bin ${AGINFER_AOT_CUBIN})
    add_executable(projection_split_cuda_test tests/projection_split_test.cc)
    target_compile_definitions(projection_split_cuda_test PRIVATE TEST_CUDA=1)
    target_compile_features(projection_split_cuda_test PRIVATE cxx_std_20)
    target_include_directories(projection_split_cuda_test PRIVATE cpp)
    target_link_libraries(projection_split_cuda_test PRIVATE aginfer_runtime CUDA::cudart)
    add_dependencies(projection_split_cuda_test projection_split_fixture aginfer_aot_kernels)
    add_test(NAME projection_split_cuda_test COMMAND projection_split_cuda_test
      ${CMAKE_CURRENT_BINARY_DIR}/projection-split.bin ${AGINFER_AOT_CUBIN})
  endif()
  if(TARGET aginfer_cublaslt_select)
    add_test(NAME cublaslt_selector_protocol_test COMMAND ${CMAKE_COMMAND} -E env
      PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_cublaslt_selector.py
      $<TARGET_FILE:aginfer_cublaslt_select>)
  endif()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rounded-attention.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_rounded_attention_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/rounded-attention.bin
    DEPENDS tests/make_rounded_attention_fixture.py src/aginfer/providers/rounded_attention.py
    VERBATIM)
  add_custom_target(rounded_attention_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/rounded-attention.bin)
  add_executable(rounded_attention_payload_test tests/rounded_attention_payload_test.cc)
  target_include_directories(rounded_attention_payload_test PRIVATE cpp)
  target_link_libraries(rounded_attention_payload_test PRIVATE aginfer_runtime)
  add_dependencies(rounded_attention_payload_test rounded_attention_fixture)
  add_test(NAME rounded_attention_payload_test COMMAND rounded_attention_payload_test
           ${CMAKE_CURRENT_BINARY_DIR}/rounded-attention.bin)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/executable-session.aim
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:${CMAKE_CURRENT_SOURCE_DIR}
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_executable_session_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/executable-session.aim
    DEPENDS tests/make_executable_session_fixture.py tests/executable_fixtures.py
            src/aginfer/executable.py src/aginfer/aim.py
    VERBATIM)
  add_custom_target(executable_session_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/executable-session.aim)
  add_executable(executable_session_test tests/executable_session_test.cc)
  target_link_libraries(executable_session_test PRIVATE aginfer_runtime)
  add_dependencies(executable_session_test executable_session_fixture)
  add_test(NAME executable_session_test COMMAND executable_session_test ${CMAKE_CURRENT_BINARY_DIR}/executable-session.aim)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/executable-plan.bin
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:${CMAKE_CURRENT_SOURCE_DIR}
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_executable_plan_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/executable-plan.bin
    DEPENDS tests/make_executable_plan_fixture.py
            tests/executable_fixtures.py src/aginfer/executable.py
            src/aginfer/lowering/command.py src/aginfer/lowering/memory.py
            src/aginfer/lowering/schedule.py
    VERBATIM)
  add_custom_target(executable_plan_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/executable-plan.bin)
  add_executable(executable_plan_test tests/executable_plan_test.cc)
  target_compile_features(executable_plan_test PRIVATE cxx_std_20)
  target_include_directories(executable_plan_test PRIVATE cpp)
  target_link_libraries(executable_plan_test PRIVATE aginfer_runtime)
  add_dependencies(executable_plan_test executable_plan_fixture)
  add_test(NAME executable_plan_test COMMAND executable_plan_test
           ${CMAKE_CURRENT_BINARY_DIR}/executable-plan.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/command-stream.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_command_stream_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/command-stream.bin
    DEPENDS tests/make_command_stream_fixture.py src/aginfer/lowering/command.py
            src/aginfer/schema.py
    VERBATIM)
  add_custom_target(command_stream_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/command-stream.bin)
  add_executable(command_stream_test tests/command_stream_test.cc)
  target_compile_features(command_stream_test PRIVATE cxx_std_20)
  target_include_directories(command_stream_test PRIVATE cpp)
  target_link_libraries(command_stream_test PRIVATE aginfer_runtime)
  add_dependencies(command_stream_test command_stream_fixture)
  add_test(NAME command_stream_test COMMAND command_stream_test
           ${CMAKE_CURRENT_BINARY_DIR}/command-stream.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cublaslt-linear.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_cublaslt_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/cublaslt-linear.bin
    DEPENDS tests/make_cublaslt_payload_fixture.py
            src/aginfer/providers/cublaslt.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(cublaslt_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cublaslt-linear.bin)
  add_executable(cublaslt_payload_test tests/cublaslt_payload_test.cc)
  target_compile_features(cublaslt_payload_test PRIVATE cxx_std_20)
  target_include_directories(cublaslt_payload_test PRIVATE cpp)
  target_link_libraries(cublaslt_payload_test PRIVATE aginfer_runtime)
  add_dependencies(cublaslt_payload_test cublaslt_payload_fixture)
  add_test(NAME cublaslt_payload_test COMMAND cublaslt_payload_test
           ${CMAKE_CURRENT_BINARY_DIR}/cublaslt-linear.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-attention.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_flashinfer_attention_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-attention.bin
    DEPENDS tests/make_flashinfer_attention_payload_fixture.py
            src/aginfer/providers/flashinfer_attention.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(flashinfer_attention_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-attention.bin)
  add_executable(flashinfer_attention_payload_test
                 tests/flashinfer_attention_payload_test.cc)
  target_compile_features(flashinfer_attention_payload_test PRIVATE cxx_std_20)
  target_include_directories(flashinfer_attention_payload_test PRIVATE cpp)
  target_link_libraries(flashinfer_attention_payload_test PRIVATE aginfer_runtime)
  add_dependencies(flashinfer_attention_payload_test
                   flashinfer_attention_payload_fixture)
  add_test(NAME flashinfer_attention_payload_test
           COMMAND flashinfer_attention_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-attention.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-prefix-attention.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_flashinfer_prefix_attention_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-prefix-attention.bin
    DEPENDS tests/make_flashinfer_prefix_attention_payload_fixture.py
            src/aginfer/providers/flashinfer_attention.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(flashinfer_prefix_attention_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-prefix-attention.bin)
  add_executable(flashinfer_prefix_attention_payload_test
                 tests/flashinfer_prefix_attention_payload_test.cc)
  target_compile_features(flashinfer_prefix_attention_payload_test PRIVATE cxx_std_20)
  target_include_directories(flashinfer_prefix_attention_payload_test PRIVATE cpp)
  target_link_libraries(flashinfer_prefix_attention_payload_test PRIVATE aginfer_runtime)
  add_dependencies(flashinfer_prefix_attention_payload_test
                   flashinfer_prefix_attention_payload_fixture)
  add_test(NAME flashinfer_prefix_attention_payload_test
           COMMAND flashinfer_prefix_attention_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/flashinfer-prefix-attention.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/adaptive-rms-norm.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_adaptive_rms_norm_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/adaptive-rms-norm.bin
    DEPENDS tests/make_adaptive_rms_norm_payload_fixture.py
            src/aginfer/providers/aot_adaptive_rms_norm.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(adaptive_rms_norm_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/adaptive-rms-norm.bin)
  add_executable(adaptive_rms_norm_payload_test
                 tests/adaptive_rms_norm_payload_test.cc)
  target_compile_features(adaptive_rms_norm_payload_test PRIVATE cxx_std_20)
  target_include_directories(adaptive_rms_norm_payload_test PRIVATE cpp)
  target_link_libraries(adaptive_rms_norm_payload_test PRIVATE aginfer_runtime)
  add_dependencies(adaptive_rms_norm_payload_test
                   adaptive_rms_norm_payload_fixture)
  add_test(NAME adaptive_rms_norm_payload_test
           COMMAND adaptive_rms_norm_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/adaptive-rms-norm.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kv-pack.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_kv_pack_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/kv-pack.bin
    DEPENDS tests/make_kv_pack_payload_fixture.py
            src/aginfer/providers/aot_kv_pack.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(kv_pack_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kv-pack.bin)
  add_executable(kv_pack_payload_test tests/kv_pack_payload_test.cc)
  target_compile_features(kv_pack_payload_test PRIVATE cxx_std_20)
  target_include_directories(kv_pack_payload_test PRIVATE cpp)
  target_link_libraries(kv_pack_payload_test PRIVATE aginfer_runtime)
  add_dependencies(kv_pack_payload_test kv_pack_payload_fixture)
  add_test(NAME kv_pack_payload_test COMMAND kv_pack_payload_test
           ${CMAKE_CURRENT_BINARY_DIR}/kv-pack.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/prefix-kv-store.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_prefix_kv_store_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/prefix-kv-store.bin
    DEPENDS tests/make_prefix_kv_store_payload_fixture.py
            src/aginfer/providers/aot_prefix_kv_store.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(prefix_kv_store_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/prefix-kv-store.bin)
  add_executable(prefix_kv_store_payload_test
                 tests/prefix_kv_store_payload_test.cc)
  target_compile_features(prefix_kv_store_payload_test PRIVATE cxx_std_20)
  target_include_directories(prefix_kv_store_payload_test PRIVATE cpp)
  target_link_libraries(prefix_kv_store_payload_test PRIVATE aginfer_runtime)
  add_dependencies(prefix_kv_store_payload_test
                   prefix_kv_store_payload_fixture)
  add_test(NAME prefix_kv_store_payload_test
           COMMAND prefix_kv_store_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/prefix-kv-store.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/prefix-input.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_prefix_input_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/prefix-input.bin
    DEPENDS tests/make_prefix_input_payload_fixture.py
            src/aginfer/providers/aot_prefix_input.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(prefix_input_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/prefix-input.bin)
  add_executable(prefix_input_payload_test
                 tests/prefix_input_payload_test.cc)
  target_compile_features(prefix_input_payload_test PRIVATE cxx_std_20)
  target_include_directories(prefix_input_payload_test PRIVATE cpp)
  target_link_libraries(prefix_input_payload_test PRIVATE aginfer_runtime)
  add_dependencies(prefix_input_payload_test prefix_input_payload_fixture)
  add_test(NAME prefix_input_payload_test COMMAND prefix_input_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/prefix-input.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/patch-projection.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_patch_projection_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/patch-projection.bin
    DEPENDS tests/make_patch_projection_payload_fixture.py
            src/aginfer/providers/patch_projection.py
            src/aginfer/providers/cublaslt.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(patch_projection_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/patch-projection.bin)
  add_executable(patch_projection_payload_test
                 tests/patch_projection_payload_test.cc)
  target_compile_features(patch_projection_payload_test PRIVATE cxx_std_20)
  target_include_directories(patch_projection_payload_test PRIVATE cpp)
  target_link_libraries(patch_projection_payload_test PRIVATE aginfer_runtime)
  add_dependencies(patch_projection_payload_test
                   patch_projection_payload_fixture)
  add_test(NAME patch_projection_payload_test
           COMMAND patch_projection_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/patch-projection.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/suffix-metadata.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_suffix_metadata_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/suffix-metadata.bin
    DEPENDS tests/make_suffix_metadata_payload_fixture.py
            src/aginfer/providers/aot_suffix_metadata.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(suffix_metadata_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/suffix-metadata.bin)
  add_executable(suffix_metadata_payload_test
                 tests/suffix_metadata_payload_test.cc)
  target_compile_features(suffix_metadata_payload_test PRIVATE cxx_std_20)
  target_include_directories(suffix_metadata_payload_test PRIVATE cpp)
  target_link_libraries(suffix_metadata_payload_test PRIVATE aginfer_runtime)
  add_dependencies(suffix_metadata_payload_test
                   suffix_metadata_payload_fixture)
  add_test(NAME suffix_metadata_payload_test
           COMMAND suffix_metadata_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/suffix-metadata.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/time-embedding.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_time_embedding_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/time-embedding.bin
    DEPENDS tests/make_time_embedding_payload_fixture.py
            src/aginfer/providers/aot_time_embedding.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(time_embedding_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/time-embedding.bin)
  add_executable(time_embedding_payload_test
                 tests/time_embedding_payload_test.cc)
  target_compile_features(time_embedding_payload_test PRIVATE cxx_std_20)
  target_include_directories(time_embedding_payload_test PRIVATE cpp)
  target_link_libraries(time_embedding_payload_test PRIVATE aginfer_runtime)
  add_dependencies(time_embedding_payload_test time_embedding_payload_fixture)
  add_test(NAME time_embedding_payload_test
           COMMAND time_embedding_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/time-embedding.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/action-slice.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_action_slice_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/action-slice.bin
    DEPENDS tests/make_action_slice_payload_fixture.py
            src/aginfer/providers/aot_action_slice.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(action_slice_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/action-slice.bin)
  add_executable(action_slice_payload_test
                 tests/action_slice_payload_test.cc)
  target_compile_features(action_slice_payload_test PRIVATE cxx_std_20)
  target_include_directories(action_slice_payload_test PRIVATE cpp)
  target_link_libraries(action_slice_payload_test PRIVATE aginfer_runtime)
  add_dependencies(action_slice_payload_test action_slice_payload_fixture)
  add_test(NAME action_slice_payload_test
           COMMAND action_slice_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/action-slice.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/layer-norm.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_layer_norm_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/layer-norm.bin
    DEPENDS tests/make_layer_norm_payload_fixture.py
            src/aginfer/providers/aot_layer_norm.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(layer_norm_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/layer-norm.bin)
  add_executable(layer_norm_payload_test tests/layer_norm_payload_test.cc)
  target_compile_features(layer_norm_payload_test PRIVATE cxx_std_20)
  target_include_directories(layer_norm_payload_test PRIVATE cpp)
  target_link_libraries(layer_norm_payload_test PRIVATE aginfer_runtime)
  add_dependencies(layer_norm_payload_test layer_norm_payload_fixture)
  add_test(NAME layer_norm_payload_test COMMAND layer_norm_payload_test
           ${CMAKE_CURRENT_BINARY_DIR}/layer-norm.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rms-norm.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_rms_norm_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/rms-norm.bin
    DEPENDS tests/make_rms_norm_payload_fixture.py
            src/aginfer/providers/aot_rms_norm.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(rms_norm_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/rms-norm.bin)
  add_executable(rms_norm_payload_test tests/rms_norm_payload_test.cc)
  target_compile_features(rms_norm_payload_test PRIVATE cxx_std_20)
  target_include_directories(rms_norm_payload_test PRIVATE cpp)
  target_link_libraries(rms_norm_payload_test PRIVATE aginfer_runtime)
  add_dependencies(rms_norm_payload_test rms_norm_payload_fixture)
  add_test(NAME rms_norm_payload_test COMMAND rms_norm_payload_test
           ${CMAKE_CURRENT_BINARY_DIR}/rms-norm.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rope.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_rope_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/rope.bin
    DEPENDS tests/make_rope_payload_fixture.py
            src/aginfer/providers/aot_rope.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(rope_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/rope.bin)
  add_executable(rope_payload_test tests/rope_payload_test.cc)
  target_compile_features(rope_payload_test PRIVATE cxx_std_20)
  target_include_directories(rope_payload_test PRIVATE cpp)
  target_link_libraries(rope_payload_test PRIVATE aginfer_runtime)
  add_dependencies(rope_payload_test rope_payload_fixture)
  add_test(NAME rope_payload_test COMMAND rope_payload_test
           ${CMAKE_CURRENT_BINARY_DIR}/rope.bin)

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vision-attention.bin
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_vision_attention_payload_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/vision-attention.bin
    DEPENDS tests/make_vision_attention_payload_fixture.py
            src/aginfer/providers/aot_vision_attention.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(vision_attention_payload_fixture
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vision-attention.bin)
  add_executable(vision_attention_payload_test
                 tests/vision_attention_payload_test.cc)
  target_compile_features(vision_attention_payload_test PRIVATE cxx_std_20)
  target_include_directories(vision_attention_payload_test PRIVATE cpp)
  target_link_libraries(vision_attention_payload_test PRIVATE aginfer_runtime)
  add_dependencies(vision_attention_payload_test
                   vision_attention_payload_fixture)
  add_test(NAME vision_attention_payload_test
           COMMAND vision_attention_payload_test
                   ${CMAKE_CURRENT_BINARY_DIR}/vision-attention.bin)

  if(AGINFER_ENABLE_CUDA)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda-kernel.bin
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
              ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_cuda_kernel_payload_fixture.py
              ${CMAKE_CURRENT_BINARY_DIR}/cuda-kernel.bin
      DEPENDS tests/make_cuda_kernel_payload_fixture.py
              src/aginfer/providers/aot_cuda.py src/aginfer/schema.py
      VERBATIM)
    add_custom_target(cuda_kernel_payload_fixture
                      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda-kernel.bin)
    add_executable(cuda_kernel_payload_test tests/cuda_kernel_payload_test.cc)
    target_compile_features(cuda_kernel_payload_test PRIVATE cxx_std_20)
    target_include_directories(cuda_kernel_payload_test PRIVATE cpp)
    target_link_libraries(cuda_kernel_payload_test PRIVATE aginfer_runtime)
    add_dependencies(cuda_kernel_payload_test cuda_kernel_payload_fixture)
    add_test(NAME cuda_kernel_payload_test COMMAND cuda_kernel_payload_test
             ${CMAKE_CURRENT_BINARY_DIR}/cuda-kernel.bin)

    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/activation.bin
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
              ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_activation_payload_fixture.py
              ${CMAKE_CURRENT_BINARY_DIR}/activation.bin
      DEPENDS tests/make_activation_payload_fixture.py
              src/aginfer/providers/aot_activation.py
              src/aginfer/providers/aot_cuda.py src/aginfer/schema.py
      VERBATIM)
    add_custom_target(activation_payload_fixture
                      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/activation.bin)
    add_executable(activation_payload_test tests/activation_payload_test.cc)
    target_compile_features(activation_payload_test PRIVATE cxx_std_20)
    target_include_directories(activation_payload_test PRIVATE cpp)
    target_link_libraries(activation_payload_test PRIVATE aginfer_runtime)
    add_dependencies(activation_payload_test activation_payload_fixture)
    add_test(NAME activation_payload_test COMMAND activation_payload_test
             ${CMAKE_CURRENT_BINARY_DIR}/activation.bin)
  endif()

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_runtime_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    DEPENDS tests/make_runtime_fixture.py src/aginfer/aim.py src/aginfer/plan.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(runtime_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test.aim)
  add_executable(c_api_test tests/c_api_test.c)
  target_compile_features(c_api_test PRIVATE c_std_11)
  target_link_libraries(c_api_test PRIVATE aginfer_runtime)
  add_dependencies(c_api_test runtime_fixture)
  add_test(NAME c_api_test COMMAND c_api_test ${CMAKE_CURRENT_BINARY_DIR}/test.aim)

  add_executable(runtime_cpp_test tests/runtime_test.cc)
  target_compile_features(runtime_cpp_test PRIVATE cxx_std_17)
  target_link_libraries(runtime_cpp_test PRIVATE aginfer_runtime)
  add_dependencies(runtime_cpp_test runtime_fixture)
  add_test(NAME runtime_cpp_test COMMAND runtime_cpp_test ${CMAKE_CURRENT_BINARY_DIR}/test.aim)

  add_executable(runtime_load_probe tests/runtime_load_probe.cc)
  target_link_libraries(runtime_load_probe PRIVATE aginfer_runtime)
  add_dependencies(runtime_load_probe runtime_fixture)
  add_test(NAME runtime_load_policy_test COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_runtime_load_policy.py
    $<TARGET_FILE:runtime_load_probe> ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    $<IF:${AGINFER_AIM_CHECKSUMS_ENABLED},verify,fast>)

  if(AGINFER_BUILD_CUDA_TESTS)
    find_package(CUDAToolkit REQUIRED)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cubin
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -cubin -arch=sm_${AGINFER_CUDA_TEST_ARCH}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_add.cu
              -o ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cubin
      DEPENDS tests/vector_add.cu
      VERBATIM)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vector_add.aim
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
              ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_cuda_fixture.py
              ${AGINFER_CUDA_TEST_ARCH}
              ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cubin
              ${CMAKE_CURRENT_BINARY_DIR}/vector_add.aim
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cubin tests/make_cuda_fixture.py
              src/aginfer/aim.py src/aginfer/plan.py
      VERBATIM)
    add_custom_target(cuda_runtime_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/vector_add.aim)
    add_executable(cuda_runtime_test tests/cuda_runtime_test.cc)
    target_link_libraries(cuda_runtime_test PRIVATE aginfer_runtime CUDA::cudart)
    add_dependencies(cuda_runtime_test cuda_runtime_fixture)
    add_test(NAME cuda_runtime_test COMMAND cuda_runtime_test ${CMAKE_CURRENT_BINARY_DIR}/vector_add.aim)
  endif()
endif()
