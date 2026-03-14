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
  # VS generator's CUDA compiler ID test .vcxproj ignores CMAKE_CUDA_FLAGS, so
  # _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH never reaches the host compiler.
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
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
  set(CMAKE_CUDA_STANDARD 17)
else()
  set(CMAKE_CUDA_STANDARD 14)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

# CCCL includes for CUDA 13+
if(IS_DIRECTORY "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  list(APPEND MUSEN_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}/cccl")
endif()

# Compile for all compute capabilities supported by the detected CUDA toolkit.
# Auto-detect on first configure; afterwards the user can edit the cache value.
if(NOT MUSEN_CUDA_ARCHS_INITIALIZED)
  execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-code
    OUTPUT_VARIABLE _gpu_arch_list
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCHALL "sm_([0-9]+)" _sm_matches "${_gpu_arch_list}")
  set(_arch_nums "")
  foreach(_sm ${_sm_matches})
    string(REGEX REPLACE "sm_" "" _num "${_sm}")
    list(APPEND _arch_nums "${_num}")
  endforeach()
  list(REMOVE_DUPLICATES _arch_nums)
  list(SORT _arch_nums COMPARE NATURAL)
  set(CMAKE_CUDA_ARCHITECTURES "${_arch_nums}" CACHE STRING "CUDA architectures" FORCE)
  set(MUSEN_CUDA_ARCHS_INITIALIZED TRUE CACHE INTERNAL "")
endif()

# CUDA 12.8+ Thrust ABI namespace uses __CUDA_ARCH_LIST__. Since Thrust headers
# leak into .cpp files, the CXX compiler must define it too to avoid linker errors.
set(MUSEN_CUDA_ARCH_DEFINITION "")
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
  set(_arch_vals "")
  foreach(_num ${CMAKE_CUDA_ARCHITECTURES})
    math(EXPR _val "${_num} * 10")
    list(APPEND _arch_vals "${_val}")
  endforeach()
  list(JOIN _arch_vals "," _cuda_arch_list)
  set(MUSEN_CUDA_ARCH_DEFINITION "__CUDA_ARCH_LIST__=${_cuda_arch_list}")
endif()
