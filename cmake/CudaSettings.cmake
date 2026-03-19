# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# CUDA detection and architecture selection.

# =============================================================================
# Allow unsupported CUDA-compiler-STL combnations.
# =============================================================================
# --allow-unsupported-compiler: bypass nvcc's host compiler version check
# _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH: bypass MSVC STL static_assert
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

if(CMAKE_GENERATOR MATCHES "Visual Studio" AND MSVC)
  # VS generator's CUDA compiler ID test .vcxproj ignores CMAKE_CUDA_FLAGS,
  # so _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH never reaches the host compiler.
  # Inject it via Directory.Build.props which MSBuild auto-imports from parent dirs.
  file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/Directory.Build.props"
"<Project>
  <ItemDefinitionGroup>
    <ClCompile>
      <PreprocessorDefinitions>_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH;%(PreprocessorDefinitions)</PreprocessorDefinitions>
    </ClCompile>
    <CudaCompile>
      <AdditionalOptions>%(AdditionalOptions) -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH </AdditionalOptions>
    </CudaCompile>
  </ItemDefinitionGroup>
</Project>
")
endif()

# =============================================================================
# Detect and setup CUDA
# =============================================================================
# Remember if the user explicitly provided CMAKE_CUDA_ARCHITECTURES via -D.
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
  set(_user_set_cuda_archs TRUE)
else()
  set(_user_set_cuda_archs FALSE)
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Workaround for CUDA overriding /Z7 flag /Zi.
# https://forums.developer.nvidia.com/t/nvcc-overrides-z7-with-zi/117009
if(CMAKE_GENERATOR MATCHES "Visual Studio" AND MSVC AND CMAKE_MSVC_DEBUG_INFORMATION_FORMAT STREQUAL "Embedded")
  file(WRITE "${CMAKE_BINARY_DIR}/Directory.Build.targets"
"<Project>
  <ItemDefinitionGroup Condition=\"'$(Configuration)' != 'Release'\">
    <CudaCompile>
      <DebugInformationFormat>OldStyle</DebugInformationFormat>
      <HostDebugInfo>false</HostDebugInfo>
    </CudaCompile>
  </ItemDefinitionGroup>
</Project>
")
endif()

# Workaround for https://github.com/NVIDIA/cccl/issues/4967
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9" AND CUDAToolkit_VERSION VERSION_LESS "13.0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --pre-include \"${CMAKE_CURRENT_LIST_DIR}/cccl_fix/CCCL4967.h\"")
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
  set(CMAKE_CUDA_STANDARD 17)
elseif(NOT MSVC)
  set(CMAKE_CUDA_STANDARD 14)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(MSVC)
  # Standard-conforming preprocessor (required by CUDA 13+).
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# Select compute capabilities to compile for:
# - User-provided -DCMAKE_CUDA_ARCHITECTURES=XX.
# - Otherwise, auto-detect all supported architectures on first configure.
# - After first configure, the user can change the cached value.
if(NOT _user_set_cuda_archs AND NOT MUSEN_CUDA_ARCHS_INITIALIZED)
  # Try --list-gpu-code for an explicit architecture list.
  execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-code
    OUTPUT_VARIABLE _gpu_arch_list
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _list_gpu_code_result
  )
  if(_list_gpu_code_result EQUAL 0 AND _gpu_arch_list)
    string(REGEX MATCHALL "sm_([0-9]+)" _sm_matches "${_gpu_arch_list}")
    set(_arch_nums "")
    foreach(_sm ${_sm_matches})
      string(REGEX REPLACE "sm_" "" _num "${_sm}")
      list(APPEND _arch_nums "${_num}")
    endforeach()
    list(REMOVE_DUPLICATES _arch_nums)
    list(SORT _arch_nums COMPARE NATURAL)
  else()
    # Fallback if --list-gpu-code is not available.
    set(_arch_nums "all")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_arch_nums}" CACHE STRING "CUDA architectures" FORCE)
  set(MUSEN_CUDA_ARCHS_INITIALIZED TRUE CACHE INTERNAL "")
endif()
