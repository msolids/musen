# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Version number generation:
# - MUSENVersion.h: generated at configure time from .h.in (main version)
# - BuildVersion.cpp: generated at build time (timestamp + git hash)

set(MUSEN_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MUSEN_GENERATED_DIR}")

# =============================================================================
# MUSENVersion.h
# =============================================================================
configure_file(
  "${CMAKE_SOURCE_DIR}/Version/MUSENVersion.h.in"
  "${MUSEN_GENERATED_DIR}/MUSENVersion.h"
  @ONLY
)

# =============================================================================
# BuildVersion.cpp
# =============================================================================
set(_gen_version_cmd ${CMAKE_COMMAND}
  -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
  -DOUTPUT_FILE=${MUSEN_GENERATED_DIR}/BuildVersion.cpp
  -DTEMPLATE_FILE=${CMAKE_SOURCE_DIR}/Version/BuildVersion.cpp.in
  -P ${CMAKE_SOURCE_DIR}/cmake/GenerateVersion.cmake
)

# Initial generation at configure time to ensure the file exists for add_library().
if(NOT EXISTS "${MUSEN_GENERATED_DIR}/BuildVersion.cpp")
  execute_process(COMMAND ${_gen_version_cmd})
endif()

# Runs at every build to update timestamp/hash.
add_custom_target(generate_build_version ALL
  COMMAND ${_gen_version_cmd}
  COMMENT "Generating build version info"
)
