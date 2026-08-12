cmake_minimum_required(VERSION 3.20)
project(AgInfer VERSION 0.1.0 LANGUAGES CXX)

add_library(aginfer_runtime STATIC cpp/runtime.cc cpp/sha256.cc)
target_compile_features(aginfer_runtime PUBLIC cxx_std_20)
target_include_directories(aginfer_runtime PUBLIC include PRIVATE cpp)
target_compile_options(aginfer_runtime PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(aginfer_runtime PRIVATE ${CMAKE_DL_LIBS})

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
            ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_runtime_fixture.py
            ${CMAKE_CURRENT_BINARY_DIR}/test.aim
    DEPENDS tests/make_runtime_fixture.py src/aginfer/aim.py src/aginfer/schema.py
    VERBATIM)
  add_custom_target(runtime_fixture DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test.aim)
  add_executable(runtime_test tests/runtime_test.cc)
  target_link_libraries(runtime_test PRIVATE aginfer_runtime)
  add_dependencies(runtime_test runtime_fixture)
  add_test(NAME runtime_test COMMAND runtime_test ${CMAKE_CURRENT_BINARY_DIR}/test.aim)
endif()
