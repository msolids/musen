# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# VS solution source filters for Solution Explorer.

function(musen_apply_source_groups SOURCES)
  foreach(SRC ${${SOURCES}})
    string(FIND "${SRC}" "${CMAKE_BINARY_DIR}" _is_generated)
    if(_is_generated EQUAL 0)
      # Generated files: place Version files in "Version", rest in "Generated Files"
      get_filename_component(_gen_name "${SRC}" NAME)
      if(_gen_name MATCHES "^(BuildVersion|MUSENVersion)")
        source_group("Version" FILES "${SRC}")
      else()
        source_group("Generated Files" FILES "${SRC}")
      endif()
    else()
      # Source files: group by directory structure
      file(RELATIVE_PATH REL_PATH "${CMAKE_SOURCE_DIR}" "${SRC}")
      get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
      string(REPLACE "/" "\\" GROUP_NAME "${REL_DIR}")
      source_group("${GROUP_NAME}" FILES "${SRC}")
    endif()
  endforeach()
endfunction()

# Place AUTOMOC, AUTOUIC, AUTORCC outputs into "Generated Files"
source_group("Generated Files" REGULAR_EXPRESSION ".*_autogen/.*")

# Place standard cmake projects into "CMake"
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

# Place third-party libraries into "Third Party"
foreach(_target zlibstatic libprotobuf libprotobuf-lite libprotoc protoc)
  if(TARGET ${_target})
    set_target_properties(${_target} PROPERTIES FOLDER "Third Party")
  endif()
endforeach()
