# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

# TODO: optional GUI
# TODO: optional GPU

CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

SET(VERSION 1.71.3)
SET(SOVERSION 1.71)

PROJECT(MUSEN VERSION ${VERSION} LANGUAGES CXX)

ENABLE_LANGUAGE(CUDA)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

INCLUDE(GNUInstallDirs)

SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

SET(lib_musen_names
  "CMusen"
  "Databases"
  "Models"
  "Modules"
  "Version"
)

# gather all sources
FOREACH(folder_name ${lib_musen_names})
  FILE(GLOB_RECURSE src   ${CMAKE_SOURCE_DIR}/${folder_name}/*.cpp ${CMAKE_SOURCE_DIR}/${folder_name}/*.cu)
  FILE(GLOB_RECURSE hdr   ${CMAKE_SOURCE_DIR}/${folder_name}/*.h   ${CMAKE_SOURCE_DIR}/${folder_name}/*.cuh)
  FILE(GLOB_RECURSE proto ${CMAKE_SOURCE_DIR}/${folder_name}/*.proto)
  SET(lib_musen_src ${lib_musen_src} ${src})
  SET(lib_musen_hdr ${lib_musen_hdr} ${hdr})
  SET(proto_files ${proto_files} ${proto})
ENDFOREACH(folder_name ${lib_musen_names})

# gather include directories
FOREACH(hdr_file ${lib_musen_hdr})
  GET_FILENAME_COMPONENT(hdr_path ${hdr_file} PATH)
  SET(hdr_paths ${hdr_paths} ${hdr_path})
ENDFOREACH()
LIST(REMOVE_DUPLICATES hdr_paths)
INCLUDE_DIRECTORIES(${hdr_paths})

# exclude model templates with dynamic libraries
SET(exclude_dirs
  "Models/Templates/ModelEF/"
  "Models/Templates/ModelPP/"
  "Models/Templates/ModelPW/"
  "Models/Templates/ModelSB/"
)
FOREACH(folder_path ${lib_musen_src})
  FOREACH(exclude_dir ${exclude_dirs})
    STRING(FIND ${folder_path} ${exclude_dir} exclude_dir_found)
    IF(NOT ${exclude_dir_found} EQUAL -1)
  	  LIST(REMOVE_ITEM lib_musen_src ${folder_path})
    ENDIF()
  ENDFOREACH(exclude_dir)
ENDFOREACH(folder_path)

# cuda
IF(${CMAKE_VERSION} VERSION_LESS "3.17.0")   
  FIND_PACKAGE(CUDA REQUIRED)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  LINK_DIRECTORIES(${CUDA_LIBRARIES})
  IF(CUDA_VERSION_MAJOR LESS 11)
    SET(CMAKE_CUDA_ARCHITECTURES "35;37;50;52;53;60;61;62;70;72;75")
  ELSE()
    # SET(CMAKE_CUDA_ARCHITECTURES "61;62;70;72;75;80;86")
    SET(CMAKE_CUDA_ARCHITECTURES "86")
  ENDIF()
ELSE() 
  FIND_PACKAGE(CUDAToolkit REQUIRED)
  INCLUDE_DIRECTORIES(${CUDAToolkit_INCLUDE_DIRS})
  LINK_DIRECTORIES(${CUDAToolkit_LIBRARY_DIR})
  IF(CUDAToolkit_VERSION_MAJOR LESS 11)
    SET(CMAKE_CUDA_ARCHITECTURES "35;37;50;52;53;60;61;62;70;72;75")
  ELSE()
    # SET(CMAKE_CUDA_ARCHITECTURES "61;62;70;72;75;80;86")
    SET(CMAKE_CUDA_ARCHITECTURES "86")
  ENDIF()
ENDIF()

# protobuf
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIRS})
LINK_DIRECTORIES(${PROTOBUF_LIBRARY_DIRS})

# generate protobuf sources
SET(proto_out_path ${CMAKE_BINARY_DIR}/ProtoGeneratedFiles)
FILE(MAKE_DIRECTORY ${proto_out_path})
FOREACH(proto ${proto_files})
  FILE(TO_NATIVE_PATH ${proto} proto_native)
  GET_FILENAME_COMPONENT(proto_path ${proto_native} DIRECTORY)
  EXECUTE_PROCESS(COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --proto_path=${proto_path} --proto_path=${CMAKE_SOURCE_DIR}/Databases/MaterialsDatabase/ --cpp_out=${proto_out_path} ${proto_native})
  GET_FILENAME_COMPONENT(proto_name_we ${proto_native} NAME_WE)
  SET(lib_musen_src ${lib_musen_src} ${proto_out_path}/${proto_name_we}.pb.cc)
  SET(lib_musen_src ${lib_musen_src} ${proto_out_path}/${proto_name_we}.pb.h)
ENDFOREACH(proto)
INCLUDE_DIRECTORIES(${proto_out_path})

# ADD_LIBRARY(libmusen_shared SHARED ${lib_musen_src})
# SET_TARGET_PROPERTIES(libmusen_shared PROPERTIES
  # VERSION ${VERSION}
  # SOVERSION ${SOVERSION}
  # PREFIX ""
  # OUTPUT_NAME "libmusen"
# )

ADD_LIBRARY(libmusen_static STATIC ${lib_musen_src})
SET_TARGET_PROPERTIES(libmusen_static PROPERTIES
  VERSION ${VERSION}
  SOVERSION ${SOVERSION}
  PREFIX ""
  OUTPUT_NAME "libmusen"
)

ADD_EXECUTABLE(CMUSEN ${proto_src} ${proto_files})

# TARGET_LINK_LIBRARIES(libmusen_shared ${PROTOBUF_LIBRARIES} ${CUDA_LIBRARIES})
TARGET_LINK_LIBRARIES(libmusen_static ${PROTOBUF_LIBRARIES} ${CUDA_LIBRARIES})
TARGET_LINK_LIBRARIES(CMUSEN libmusen_static)

# TARGET_COMPILE_DEFINITIONS(libmusen_shared PRIVATE INSTALL_DOCS_PATH="${INSTALL_DOCS_PATH}")
TARGET_COMPILE_DEFINITIONS(libmusen_static PRIVATE INSTALL_DOCS_PATH="${INSTALL_DOCS_PATH}")

SET(INSTALL_PATH_ROOT ${CMAKE_INSTALL_PREFIX}/MUSEN/)

INSTALL(TARGETS CMUSEN DESTINATION ${INSTALL_PATH_ROOT})
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Databases/" DESTINATION ${INSTALL_PATH_ROOT}/Databases)
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Examples/"  DESTINATION ${INSTALL_PATH_ROOT}/Examples)
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Licenses/"  DESTINATION ${INSTALL_PATH_ROOT}/Licenses)
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Documentation/"             DESTINATION ${INSTALL_PATH_ROOT}/Documentation)
INSTALL(FILES     "${CMAKE_SOURCE_DIR}/LICENSE"                    DESTINATION ${INSTALL_PATH_ROOT})

# INSTALL(TARGETS libmusen_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
# INSTALL(TARGETS libmusen_static DESTINATION ${CMAKE_INSTALL_LIBDIR})

ADD_SUBDIRECTORY("${CMAKE_SOURCE_DIR}/MusenGUI")
