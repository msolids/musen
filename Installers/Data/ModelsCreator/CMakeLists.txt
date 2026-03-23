# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
# See LICENSE file for license and warranty information.

# Standalone CMake project for building custom MUSEN model DLLs/shared libraries.
# Prerequisites: C++ compiler, CUDA Toolkit, CMake 3.23+.
#
# Windows:
#   cmake -B build -G "Visual Studio 17 2022"
#   cmake --build build --config Release
#
# Linux:
#   cmake -B build
#   cmake --build build
#
# Then add path to the resulting .dll/.so files to the MUSEN's Models Manager.

cmake_minimum_required(VERSION 3.23)
project(MUSEN_ModelsCreator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# Place .h and .cuh files into "Header Files" filter in VS Solution Explorer.
source_group("Header Files" REGULAR_EXPRESSION "\\.(h|cuh)$")

# --- CUDA setup ---------------------------------------------------------------
# Allow unsupported CUDA-compiler-STL combnations.
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

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Compile for all compute capabilities supported by the detected CUDA toolkit.
# Auto-detect on first configure; afterwards the user can edit the cache value.
if(NOT _MUSEN_MC_ARCHS_INITIALIZED)
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
  set(_MUSEN_MC_ARCHS_INITIALIZED TRUE CACHE INTERNAL "")
endif()

# --- VS debugger paths --------------------------------------------------------
if(MSVC)
  set(_MUSEN_DEBUG_EXE "${CMAKE_SOURCE_DIR}/DebugRun/MUSEN.exe")
  set(_MUSEN_DEBUG_DIR "${CMAKE_SOURCE_DIR}/DebugRun")
  set(_MUSEN_RELEASE_EXE "C:/Program Files/MUSEN/MUSEN.exe")
  set(_MUSEN_RELEASE_DIR "C:/Program Files/MUSEN")
  set(_MUSEN_DBG_CMD "$<IF:$<CONFIG:Debug>,${_MUSEN_DEBUG_EXE},${_MUSEN_RELEASE_EXE}>")
  set(_MUSEN_DBG_DIR "$<IF:$<CONFIG:Debug>,${_MUSEN_DEBUG_DIR},${_MUSEN_RELEASE_DIR}>")
endif()

# --- MUSEN API (static library) -----------------------------------------------
file(GLOB _api_headers "MUSEN_API/*.h" "MUSEN_API/*.cuh")
add_library(MUSEN_API STATIC
  MUSEN_API/AbstractDEMModel.cpp
  MUSEN_API/SceneTypes.cpp
  ${_api_headers}
)
target_include_directories(MUSEN_API PUBLIC MUSEN_API)
target_link_libraries(MUSEN_API PUBLIC CUDA::cudart_static)
target_compile_definitions(MUSEN_API PUBLIC
  _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
)
set_target_properties(MUSEN_API PROPERTIES FOLDER "API")
if(MSVC)
  set_target_properties(MUSEN_API PROPERTIES
    VS_DEBUGGER_COMMAND           "${_MUSEN_DBG_CMD}"
    VS_DEBUGGER_WORKING_DIRECTORY "${_MUSEN_DBG_DIR}"
  )
  set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT MUSEN_API)
endif()

# --- Helper function to add a model shared library ----------------------------
# Optional EXCLUDE_FROM_DEFAULT_BUILD after the three required arguments.
function(add_musen_model name dir folder)
  file(GLOB _src_cpp "${dir}/*.cpp")
  file(GLOB _src_cu  "${dir}/*.cu")
  file(GLOB _src_h   "${dir}/*.h" "${dir}/*.cuh")
  add_library(${name} SHARED ${_src_cpp} ${_src_cu} ${_src_h})
  target_link_libraries(${name} PRIVATE MUSEN_API)
  set_target_properties(${name} PROPERTIES FOLDER "${folder}")
  if(ARGN)
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  endif()
  if(MSVC)
    # cudart_static is always /MT: suppress its conflicting runtime in /MD builds
    target_link_options(${name} PRIVATE /NODEFAULTLIB:LIBCMT)
  endif()
endfunction()

# --- Template models -------------------------
add_musen_model(ModelPP Models/ModelPP "Models")
add_musen_model(ModelPW Models/ModelPW "Models")
add_musen_model(ModelSB Models/ModelSB "Models")
add_musen_model(ModelEF Models/ModelEF "Models")

# --- Example models -----------
add_musen_model(HertzMindlin     ExampleModels/ParticleParticle/HertzMindlin "Examples" EXCLUDE_FROM_DEFAULT_BUILD)
add_musen_model(PWHertzMindlin   ExampleModels/ParticleWall/PWHertzMindlin   "Examples" EXCLUDE_FROM_DEFAULT_BUILD)
add_musen_model(BondModelElastic ExampleModels/SolidBonds/BondModelElastic   "Examples" EXCLUDE_FROM_DEFAULT_BUILD)
add_musen_model(ViscousField     ExampleModels/ExternalForce/ViscousField    "Examples" EXCLUDE_FROM_DEFAULT_BUILD)

# Mark example and API source files as read-only.
file(GLOB_RECURSE _readonly_files "ExampleModels/*" "MUSEN_API/*")
file(CHMOD ${_readonly_files} PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
