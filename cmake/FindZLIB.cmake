# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Custom FindZLIB override.
# When both zlib and protobuf are built from source (FetchContent), protobuf's
# internal find_package(ZLIB) would fail because CMake's built-in FindZLIB
# doesn't recognize the FetchContent-created ZLIB::ZLIB target.
# This override detects the existing target and reports it as found.

if(TARGET ZLIB::ZLIB)
  set(ZLIB_FOUND TRUE)
  return()
endif()

# Delegate to the built-in module
include(${CMAKE_ROOT}/Modules/FindZLIB.cmake)
