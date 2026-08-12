cmake_minimum_required(VERSION 3.20)
project(AgInfer VERSION 0.1.0 LANGUAGES CXX)

add_library(aginfer_runtime STATIC
  cpp/cuda_driver.cc
  cpp/plan.cc
  cpp/runtime.cc
  cpp/sha256.cc)
target_compile_features(aginfer_runtime PUBLIC cxx_std_20)
target_include_directories(aginfer_runtime PUBLIC include PRIVATE cpp)
target_compile_options(aginfer_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(aginfer_runtime PRIVATE ${CMAKE_DL_LIBS})

include(CTest)
option(AGINFER_BUILD_CUDA_TESTS "Build the real sm89 CUDA execution test" OFF)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_runtime_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    DEPENDS tests/make_runtime_fixture.py src/aginfer/aim.py src/aginfer/plan.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(runtime_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test.aim)
  add_executable(runtime_test tests/runtime_test.cc)
  target_link_libraries(runtime_test PRIVATE aginfer_runtime)
  add_dependencies(runtime_test runtime_fixture)
  add_test(NAME runtime_test COMMAND runtime_test ${CMAKE_CURRENT_BINARY_DIR}/test.aim)

  if(AGINFER_BUILD_CUDA_TESTS)
    find_package(CUDAToolkit REQUIRED)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cubin
      COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -cubin -arch=sm_89
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/vector_add.cu
              -o ${CMAKE_CURRENT_BINARY_DIR}/vector_add.cubin
      DEPENDS tests/vector_add.cu
      VERBATIM)
    add_custom_command(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vector_add.aim
      COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
              ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_cuda_fixture.py
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
