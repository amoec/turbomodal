cmake_minimum_required(VERSION 3.20)

# macOS: must be set BEFORE project() so the compiler is configured correctly.
# Targets 11.0 (Big Sur, first Apple Silicon release) to avoid newer libc++
# symbols (e.g. __hash_memory from LLVM 19) that break conda/older runtimes.
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
endif()

project(turbomodal VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure RPATH is set for test executables to find shared libraries (Unix only)
if(NOT WIN32)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

# MSVC-specific settings
if(MSVC)
  add_compile_options(/utf-8 /W3 /EHsc)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
endif()

option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_VALIDATION_TESTS "Build slow validation tests (requires mesh files)" OFF)
option(USE_OPENMP "Enable OpenMP parallelism" ON)

# --- Eigen 3.4.0 via FetchContent ---
# System has Eigen 5.0.1 which is incompatible with Spectra.
# Download Eigen 3.4.0 locally to ensure compatibility.
include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

# --- Spectra (header-only, git submodule) ---
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/external/spectra/include/Spectra")
  message(FATAL_ERROR "Spectra not found. Run: git submodule update --init")
endif()
add_library(spectra INTERFACE)
target_include_directories(spectra INTERFACE
  "${CMAKE_SOURCE_DIR}/external/spectra/include"
)

# --- OpenMP (optional) ---
if(USE_OPENMP)
  find_package(OpenMP)
endif()

# --- Core static library ---
add_library(turbomodal_core STATIC
  src/material.cpp
  src/element.cpp
  src/mesh.cpp
  src/assembler.cpp
  src/rotating_effects.cpp
  src/modal_solver.cpp
  src/cyclic_solver.cpp
  src/added_mass.cpp
  src/damping.cpp
  src/forced_response.cpp
  src/mistuning.cpp
  src/mode_identification.cpp
)
target_include_directories(turbomodal_core PUBLIC include)
target_link_libraries(turbomodal_core PUBLIC Eigen3::Eigen spectra)
if(OpenMP_CXX_FOUND)
  target_link_libraries(turbomodal_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- Unit tests ---
if(BUILD_TESTS)
  enable_testing()

  # Fetch GTest via FetchContent (self-contained, works on all platforms)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(test_material tests/test_material.cpp)
  target_link_libraries(test_material turbomodal_core GTest::gtest_main)
  add_test(NAME MaterialTests COMMAND test_material)

  add_executable(test_element tests/test_element.cpp)
  target_link_libraries(test_element turbomodal_core GTest::gtest_main)
  add_test(NAME ElementTests COMMAND test_element)

  add_executable(test_mesh tests/test_mesh.cpp)
  target_link_libraries(test_mesh turbomodal_core GTest::gtest_main)
  target_compile_definitions(test_mesh PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
  add_test(NAME MeshTests COMMAND test_mesh)

  add_executable(test_assembler tests/test_assembler.cpp)
  target_link_libraries(test_assembler turbomodal_core GTest::gtest_main)
  target_compile_definitions(test_assembler PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
  add_test(NAME AssemblerTests COMMAND test_assembler)

  add_executable(test_solver tests/test_solver.cpp)
  target_link_libraries(test_solver turbomodal_core GTest::gtest_main)
  target_compile_definitions(test_solver PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
  add_test(NAME SolverTests COMMAND test_solver)

  add_executable(test_cyclic tests/test_cyclic.cpp)
  target_link_libraries(test_cyclic turbomodal_core GTest::gtest_main)
  target_compile_definitions(test_cyclic PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
  add_test(NAME CyclicTests COMMAND test_cyclic)

  add_executable(test_added_mass tests/test_added_mass.cpp)
  target_link_libraries(test_added_mass turbomodal_core GTest::gtest_main)
  add_test(NAME AddedMassTests COMMAND test_added_mass)

  add_executable(test_rotating tests/test_rotating.cpp)
  target_link_libraries(test_rotating turbomodal_core GTest::gtest_main)
  target_compile_definitions(test_rotating PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
  add_test(NAME RotatingTests COMMAND test_rotating)

  add_executable(test_damping tests/test_damping.cpp)
  target_link_libraries(test_damping turbomodal_core GTest::gtest_main)
  add_test(NAME DampingTests COMMAND test_damping)

  add_executable(test_forced_response tests/test_forced_response.cpp)
  target_link_libraries(test_forced_response turbomodal_core GTest::gtest_main)
  add_test(NAME ForcedResponseTests COMMAND test_forced_response)

  add_executable(test_mistuning tests/test_mistuning.cpp)
  target_link_libraries(test_mistuning turbomodal_core GTest::gtest_main)
  add_test(NAME MistuningTests COMMAND test_mistuning)

  add_executable(test_mode_identification tests/test_mode_identification.cpp)
  target_link_libraries(test_mode_identification turbomodal_core GTest::gtest_main)
  target_compile_definitions(test_mode_identification PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
  add_test(NAME ModeIdentificationTests COMMAND test_mode_identification)

  if(BUILD_VALIDATION_TESTS)
    add_executable(test_validation tests/test_validation.cpp)
    target_link_libraries(test_validation turbomodal_core GTest::gtest_main)
    target_compile_definitions(test_validation PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
    add_test(NAME ValidationTests COMMAND test_validation)

    add_executable(test_integration tests/test_integration.cpp)
    target_link_libraries(test_integration turbomodal_core GTest::gtest_main)
    target_compile_definitions(test_integration PRIVATE
      TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data")
    add_test(NAME IntegrationTests COMMAND test_integration)
  endif()
endif()

# --- Python bindings ---
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core src/python_bindings.cpp src/compat_hash.c)
  target_link_libraries(_core PRIVATE turbomodal_core)
  if(DEFINED SKBUILD)
    install(TARGETS _core DESTINATION turbomodal)
  else()
    install(TARGETS _core DESTINATION .)
  endif()
endif()
