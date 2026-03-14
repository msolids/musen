# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Proto files compilation: generates .pb.cc/.pb.h at build time.

function(musen_protobuf_generate OUT_SOURCES)
  set(PROTO_OUT_DIR "${CMAKE_BINARY_DIR}/ProtoGeneratedFiles")
  file(MAKE_DIRECTORY "${PROTO_OUT_DIR}")

  # Find all .proto files
  file(GLOB_RECURSE PROTO_FILES
    "${CMAKE_SOURCE_DIR}/Databases/*.proto"
    "${CMAKE_SOURCE_DIR}/Modules/*.proto"
  )

  # Determine protoc executable
  if(TARGET protobuf::protoc)
    set(_PROTOC_CMD protobuf::protoc)
  elseif(Protobuf_PROTOC_EXECUTABLE)
    set(_PROTOC_CMD "${Protobuf_PROTOC_EXECUTABLE}")
  else()
    message(FATAL_ERROR "protoc not found. Install protobuf or let FetchContent build it.")
  endif()

  set(_generated_files "")
  foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_DIR  "${PROTO_FILE}" DIRECTORY)
    get_filename_component(PROTO_NAME "${PROTO_FILE}" NAME_WE)

    set(_out_cc "${PROTO_OUT_DIR}/${PROTO_NAME}.pb.cc")
    set(_out_h  "${PROTO_OUT_DIR}/${PROTO_NAME}.pb.h")

    add_custom_command(
      OUTPUT "${_out_cc}" "${_out_h}"
      COMMAND ${_PROTOC_CMD}
        --proto_path=${PROTO_DIR}
        --proto_path=${CMAKE_SOURCE_DIR}/Databases/MaterialsDatabase/
        --cpp_out=${PROTO_OUT_DIR}
        ${PROTO_FILE}
      DEPENDS ${PROTO_FILE}
      COMMENT "Compiling proto file: ${PROTO_FILE}"
      VERBATIM
    )
    list(APPEND _generated_files "${_out_cc}" "${_out_h}")
  endforeach()

  set(${OUT_SOURCES} ${_generated_files} PARENT_SCOPE)
  set(PROTO_OUT_DIR "${PROTO_OUT_DIR}" PARENT_SCOPE)
endfunction()
