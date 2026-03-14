# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Generates installer_info.ini with git branch for Inno Setup installer in build-time.

set(_branch "unknown")
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
      COMMAND ${GIT_EXECUTABLE} symbolic-ref --short HEAD
      WORKING_DIRECTORY "${SOURCE_DIR}"
      OUTPUT_VARIABLE _branch
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()
endif()

file(WRITE "${OUTPUT_FILE}" "[Version]\nBranch=${_branch}\n")
