# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Windows installer target using Inno Setup.

if(NOT WIN32 OR NOT MUSEN_BUILD_INSTALLER)
  return()
endif()

# Find Inno Setup compiler
set(ISCC_PATH "${CMAKE_SOURCE_DIR}/Installers/Compiler/ISCC.exe")
if(NOT EXISTS "${ISCC_PATH}")
  find_program(ISCC_PATH ISCC)
endif()
if(NOT ISCC_PATH)
  message(WARNING "Inno Setup compiler (ISCC.exe) not found - 'installer' target will not be available")
  return()
endif()

set(_ISS_SCRIPT    "${CMAKE_SOURCE_DIR}/Installers/Scripts/Main.iss")
set(_ISS_INFO_INI  "${MUSEN_GENERATED_DIR}/installer_info.ini")

# Qt path for the installer's QtLibs.iss (MUSEN_QT_PREFIX set in Dependencies.cmake)
set(_ISS_QT_DEFINE "")
if(MUSEN_QT_PREFIX)
  set(_ISS_QT_DEFINE "/DQtPath=${MUSEN_QT_PREFIX}")
endif()

# Custom target: build both Release and Debug, then run ISCC.
# The installer bundles Release binaries + Debug MUSEN.exe (for ModelsCreator).
add_custom_target(installer
  # Build Release
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Release
  # Build Debug (only MUSEN needed, but building all is simpler and safer)
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config Debug
  # Generate installer_info.ini with git branch
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
    -DOUTPUT_FILE=${_ISS_INFO_INI}
    -P ${CMAKE_SOURCE_DIR}/cmake/GenerateInstallerData.cmake
  # Run Inno Setup compiler
  COMMAND "${ISCC_PATH}"
    "/DReleaseDir=${CMAKE_BINARY_DIR}/Release"
    "/DDebugDir=${CMAKE_BINARY_DIR}/Debug"
    "/DInstallerInfoFile=${_ISS_INFO_INI}"
    ${_ISS_QT_DEFINE}
    "${_ISS_SCRIPT}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Building Windows installer"
  VERBATIM
)
