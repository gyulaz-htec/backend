# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)

project(tritonbackend LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend utilities" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm MALI GPU support in backend utilities" OFF)
option(TRITON_ENABLE_ROCM "Enable AMD GPU support in backend utilities" OFF)
option(TRITON_ENABLE_MIGRAPHX "Enable AMD GPU support with MIGraphX in backend utilities" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend utilities" ON)
# Default OFF unless backend explicitly request to use provided implementation
option(TRITON_ENABLE_MEMORY_TRACKER "Include device memory tracker in backend utilities" OFF)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

# Brian
if(TRITON_ENABLE_ROCM OR TRITON_ENABLE_MIGRAPHX)
  include_directories(/opt/rocm/include)
endif()

#
# Setting C++ standard
#
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(TRITON_ENABLE_MEMORY_TRACKER AND NOT TRITON_ENABLE_GPU)
  message(WARNING "TRITON_ENABLE_MEMORY_TRACKER=ON requires TRITON_ENABLE_GPU=ON, TRITON_ENABLE_MEMORY_TRACKER will be disable")
  set(TRITON_ENABLE_MEMORY_TRACKER OFF CACHE BOOL "Device memory tracker disabled" FORCE)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  # GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_REPOSITORY https://github.com/gyulaz-htec/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
  find_package(CUDAToolkit REQUIRED)
  find_package(CUDA REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++${TRITON_MIN_CXX_STANDARD})

  if(CUDA_VERSION VERSION_GREATER "10.1" OR CUDA_VERSION VERSION_EQUAL "10.1")
    add_definitions(-DTRITON_ENABLE_CUDA_GRAPH=1)
  else()
    message(WARNING "CUDA ${CUDA_VERSION} does not support CUDA graphs.")
  endif()
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ROCM})
  find_package(hip REQUIRED)
  find_package(HIP)
  enable_language(HIP)
endif() # TRITON_ENABLE_ROCM

#
# Backend library containing useful source and utilities
#

if (NOT ${TRITON_ENABLE_ROCM})
  set(SRC_FILES
    "src/backend_common.cc"
    "src/backend_input_collector.cc"
    "src/backend_memory.cc"
    "src/backend_model_instance.cc"
    "src/backend_model.cc"
    "src/backend_output_responder.cc"
  )

  if(${TRITON_ENABLE_GPU})
    set(SRC_FILES ${SRC_FILES} "src/kernel.h")
    if(${TRITON_ENABLE_MEMORY_TRACKER})
      set(SRC_FILES ${SRC_FILES} "src/device_memory_tracker.cc")
    endif() # TRITON_ENABLE_MEMORY_TRACKER
  endif() # TRITON_ENABLE_GPU

  add_library(
    triton-backend-utils
    ${SRC_FILES}
  )
endif()

if(${TRITON_ENABLE_GPU})
  set(HOST_COMPILER_FLAGS "")
  if (WIN32)
    set(HOST_COMPILER_FLAGS "/MD")
  else()
    set(HOST_COMPILER_FLAGS "-fPIC")
  endif()

  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  cuda_add_library(
    kernel_library_new
    src/kernel.cu src/kernel.h
    OPTIONS -arch compute_53
    OPTIONS -code compute_53,sm_53,sm_60,sm_61,sm_62,sm_70,sm_72,sm_75
    OPTIONS -Xcompiler ${HOST_COMPILER_FLAGS}
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ROCM})
  set(TRITON_ROCM_HOME /opt/rocm)

  list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/llvm $ENV{ROCM_PATH} $ENV{HIP_PATH})
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hcc /opt/rocm/hip /opt/rocm)

  if (NOT CMAKE_HIP_COMPILER)
    set(CMAKE_HIP_COMPILER "${TRITON_ROCM_HOME}/llvm/bin/clang++")
  endif()

  if (NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100;gfx1101;gfx940;gfx941;gfx942")
  endif()

  file(GLOB rocm_cmake_components ${TRITON_ROCM_HOME}/lib/cmake/*)
  list(APPEND CMAKE_PREFIX_PATH ${rocm_cmake_components})

  # replicate strategy used by pytorch to get ROCM_VERSION
  file(READ "${TRITON_ROCM_HOME}/.info/version-dev" ROCM_VERSION_DEV_RAW)
  string(REGEX MATCH "^([0-9]+)\.([0-9]+)\.([0-9]+)-.*$" ROCM_VERSION_DEV_MATCH ${ROCM_VERSION_DEV_RAW})
  if (ROCM_VERSION_DEV_MATCH)
    set(ROCM_VERSION_DEV_MAJOR ${CMAKE_MATCH_1})
    set(ROCM_VERSION_DEV_MINOR ${CMAKE_MATCH_2})
    set(ROCM_VERSION_DEV_PATCH ${CMAKE_MATCH_3})
    set(ROCM_VERSION_DEV "${ROCM_VERSION_DEV_MAJOR}.${ROCM_VERSION_DEV_MINOR}.${ROCM_VERSION_DEV_PATCH}")
    math(EXPR ROCM_VERSION_DEV_INT "(${ROCM_VERSION_DEV_MAJOR}*10000) + (${ROCM_VERSION_DEV_MINOR}*100) + ${ROCM_VERSION_DEV_PATCH}")
  endif()
  message("\n***** ROCm version from ${TRITON_ROCM_HOME}/.info/version-dev ****\n")
  message("ROCM_VERSION_DEV: ${ROCM_VERSION_DEV}")
  message("ROCM_VERSION_DEV_MAJOR: ${ROCM_VERSION_DEV_MAJOR}")
  message("ROCM_VERSION_DEV_MINOR: ${ROCM_VERSION_DEV_MINOR}")
  message("ROCM_VERSION_DEV_PATCH: ${ROCM_VERSION_DEV_PATCH}")
  message("ROCM_VERSION_DEV_INT:   ${ROCM_VERSION_DEV_INT}")
  message("\n***** HIP LANGUAGE CONFIG INFO ****\n")
  message("CMAKE_HIP_COMPILER:      ${CMAKE_HIP_COMPILER}")
  message("CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
  message("CMAKE_HIP_FLAGS:         ${CMAKE_HIP_FLAGS}")

  if (NOT TRITON_HIPIFY_PERL)
    find_path(HIPIFY_PERL_PATH
      NAMES hipify-perl
      HINTS
      ${TRITON_BUILD_ROCM_HOME}/bin
      ${TRITON_BUILD_ROCM_HOME}/hip/bin)
    if (HIPIFY_PERL_PATH-NOTFOUND)
      MESSAGE(FATAL_ERROR "hipify-perl not found")
    endif()
    set(TRITON_HIPIFY_PERL ${HIPIFY_PERL_PATH}/hipify-perl)
  endif() #TRITON_HIPIFY_PERL

  include(triton_rocm_hipify.cmake)
  set(REPO_ROOT ${PROJECT_SOURCE_DIR})
  hipify(src "" triton_hipified_cc_srcs triton_hipified_cu_srcs)

  add_custom_target(generated_hipified_files DEPENDS
    ${triton_hipified_cc_srcs}
    ${triton_hipified_cu_srcs})

  # auto_set_source_files_hip_language(triton_hipified_cu_srcs)

  add_library(
    triton-backend-utils
    ${triton_hipified_cc_srcs}
  )

  add_library(
    kernel_library_new
    ${triton_hipified_cu_srcs}
  )
  set_property(TARGET kernel_library_new PROPERTY POSITION_INDEPENDENT_CODE ON)
endif() # TRITON_ENABLE_ROCM

add_library(
  TritonBackend::triton-backend-utils ALIAS triton-backend-utils
)

if(${TRITON_ENABLE_ROCM})
  set(INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/amdgpu/include")
  message("*** Include path for ROCM files: ${INCLUDE_PATH}")
  target_include_directories(
    triton-backend-utils
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${INCLUDE_PATH}>
    PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/src
  )
else()
  target_include_directories(
    triton-backend-utils
    PUBLIC
      $<INSTALL_INTERFACE:include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message("Using MSVC as compiler, default target on Windows 10. "
      "If the target system is not Windows 10, please update _WIN32_WINNT "
      "to corresponding value.")
endif()

target_compile_features(triton-backend-utils PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-backend-utils
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Werror -Wno-unused-result>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

# TRITON_ENABLE_GPU exposed in header so set PUBLIC
if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    triton-backend-utils
    PUBLIC TRITON_ENABLE_GPU=1
  )
  if(${TRITON_ENABLE_MEMORY_TRACKER})
    target_compile_definitions(
      triton-backend-utils
      PUBLIC TRITON_ENABLE_MEMORY_TRACKER=1
    )
  endif() # TRITON_ENABLE_MEMORY_TRACKER
endif() # TRITON_ENABLE_GPU

# TRITON_ENABLE_MALI_GPU exposed in header so set PUBLIC
if(${TRITON_ENABLE_MALI_GPU})
target_compile_definitions(
  triton-backend-utils
  PUBLIC TRITON_ENABLE_MALI_GPU=1
)
endif() # TRITON_ENABLE_MALI_GPU

# TRITON_ENABLE_ROCM exposed in header so set PUBLIC
if(${TRITON_ENABLE_ROCM})
target_compile_definitions(
  triton-backend-utils
  PUBLIC TRITON_ENABLE_ROCM=1
)
endif() # TRITON_ENABLE_ROCM

# TRITON_ENABLE_STATS exposed in header so set PUBLIC
if(${TRITON_ENABLE_STATS})
target_compile_definitions(
  triton-backend-utils
  PUBLIC TRITON_ENABLE_STATS=1
)
endif() # TRITON_ENABLE_STATS

set_target_properties(
  triton-backend-utils PROPERTIES
  WINDOWS_EXPORT_ALL_SYMBOLS TRUE
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME tritonbackendutils
)

target_link_libraries(
  triton-backend-utils
  PUBLIC
    triton-core-backendapi         # from repo-core
    triton-core-serverapi          # from repo-core
    triton-common-async-work-queue # from repo-common
    triton-common-json             # from repo-common
)

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    triton-backend-utils
    PUBLIC
      CUDA::cudart
    PRIVATE
      kernel_library_new
  )
  if(${TRITON_ENABLE_MEMORY_TRACKER})
    target_link_libraries(
      triton-backend-utils
      PUBLIC
        CUDA::cupti
    )
  endif() # TRITON_ENABLE_MEMORY_TRACKER
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ROCM})
  target_link_libraries(
    triton-backend-utils
    PUBLIC
      hip::host
    PRIVATE
      kernel_library_new
  )
endif() #TRITON_ENABLE_ROCM

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonBackend)

install(
  TARGETS
    triton-backend-utils
  EXPORT
    triton-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(${TRITON_ENABLE_GPU})
  install(
    TARGETS
      kernel_library_new
    EXPORT
      triton-backend-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_ROCM})
  install(
    TARGETS
      kernel_library_new
    EXPORT
      triton-backend-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif() # TRITON_ENABLE_GPU

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/include
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT
    triton-backend-targets
  FILE
    TritonBackendTargets.cmake
  NAMESPACE
    TritonBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonBackendTargets.cmake
  NAMESPACE TritonBackend::
)

export(PACKAGE TritonBackend)
