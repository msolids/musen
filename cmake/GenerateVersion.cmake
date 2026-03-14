# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Build-time script: generates BuildVersion.cpp with current timestamp and git hash.
# Called via add_custom_target in VersionSetup.cmake.
# Only regenerates when the git hash changes to avoid unnecessary recompiles.

# --- Git hash and branch ---
set(BUILD_HASH "")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-work-tree
    WORKING_DIRECTORY "${SOURCE_DIR}"
    OUTPUT_VARIABLE _git_inside
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result
  )
  if(_git_result EQUAL 0 AND _git_inside STREQUAL "true")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY "${SOURCE_DIR}"
      OUTPUT_VARIABLE _git_hash
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY "${SOURCE_DIR}"
      OUTPUT_VARIABLE _git_branch
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_git_hash AND _git_branch)
      set(BUILD_HASH "${_git_hash}.${_git_branch}")
    endif()
  endif()
endif()

# --- Check if git state changed since last generation ---
set(_cache_file "${OUTPUT_FILE}.last_hash")
set(_needs_regeneration TRUE)
if(EXISTS "${_cache_file}" AND EXISTS "${OUTPUT_FILE}")
  file(READ "${_cache_file}" _last_hash)
  string(STRIP "${_last_hash}" _last_hash)
  if(_last_hash STREQUAL BUILD_HASH)
    set(_needs_regeneration FALSE)
  endif()
endif()

# --- Only regenerate when hash changed ---
# Uses copy_if_different so BuildVersion.cpp only actually changes when content differs.
if(_needs_regeneration)
  string(TIMESTAMP BUILD_TIME "%y%m%d.%H%M%S")
  configure_file("${TEMPLATE_FILE}" "${OUTPUT_FILE}.tmp" @ONLY)
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUTPUT_FILE}.tmp" "${OUTPUT_FILE}")
  file(REMOVE "${OUTPUT_FILE}.tmp")
  file(WRITE "${_cache_file}" "${BUILD_HASH}")
endif()
