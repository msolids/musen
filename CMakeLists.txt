# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

# TODO: optional GUI
# TODO: optional GPU

CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

SET(VERSION 1.71.3)
SET(SOVERSION 1.71)

PROJECT(MUSEN VERSION ${VERSION} LANGUAGES CXX)

OPTION(BUILD_GUI "Build a version with graphical user interface" ON)
OPTION(BUILD_CLI "Build a version with command line interface" ON)

ENABLE_LANGUAGE(CUDA)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CUDA_STANDARD 14)
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
  SET(cuda_package_name "CUDA")
  SET(cuda_include_dirs_name "CUDA_INCLUDE_DIRS")
  SET(cuda_link_directories_name "CUDA_LIBRARIES")
  SET(cuda_version_major_name "CUDA_VERSION_MAJOR")
  SET(cuda_version_minor_name "CUDA_VERSION_MINOR")
ELSE()
  SET(cuda_package_name "CUDAToolkit")
  SET(cuda_include_dirs_name "CUDAToolkit_INCLUDE_DIRS")
  SET(cuda_link_directories_name "CUDAToolkit_LIBRARY_DIR")
  SET(cuda_version_major_name "CUDAToolkit_VERSION_MAJOR")
  SET(cuda_version_minor_name "CUDAToolkit_VERSION_MINOR")
ENDIF()
SET(cuda_cc_9    "30;32;35;37;50;52;53;60;61;62;70")
SET(cuda_cc_10   "30;32;35;37;50;52;53;60;61;62;70;72;75")
SET(cuda_cc_11_0 "50;52;53;60;61;62;70;72;75;80")
SET(cuda_cc_11_1 "50;52;53;60;61;62;70;72;75;80;86")
SET(cuda_cc_11_5 "50;52;53;60;61;62;70;72;75;80;86;87")
SET(cuda_cc_11_8 "50;52;53;60;61;62;70;72;75;80;86;87;89;90")
FIND_PACKAGE(${cuda_package_name} REQUIRED)
INCLUDE_DIRECTORIES(${${cuda_include_dirs_name}})
LINK_DIRECTORIES(${cuda_link_directories_name})
IF(${cuda_version_major_name} LESS 10)
  SET(CMAKE_CUDA_ARCHITECTURES ${cuda_cc_9})
ELSEIF(${cuda_version_major_name} LESS 11)
  SET(CMAKE_CUDA_ARCHITECTURES ${cuda_cc_10})
ELSEIF(${cuda_version_minor_name} LESS 1)
  SET(CMAKE_CUDA_ARCHITECTURES ${cuda_cc_11_0})
ELSEIF(${cuda_version_minor_name} LESS 5)
  SET(CMAKE_CUDA_ARCHITECTURES ${cuda_cc_11_1})
ELSEIF(${cuda_version_minor_name} LESS 8)
  SET(CMAKE_CUDA_ARCHITECTURES ${cuda_cc_11_5})
ELSE()
  SET(CMAKE_CUDA_ARCHITECTURES ${cuda_cc_11_8})
ENDIF()

# protobuf
FIND_PACKAGE(Protobuf REQUIRED)
INCLUDE_DIRECTORIES(${Protobuf_INCLUDE_DIRS})

# generate protobuf sources
SET(proto_out_path ${CMAKE_BINARY_DIR}/ProtoGeneratedFiles)
FILE(MAKE_DIRECTORY ${proto_out_path})
FOREACH(proto ${proto_files})
  FILE(TO_NATIVE_PATH ${proto} proto_native)
  GET_FILENAME_COMPONENT(proto_path ${proto_native} DIRECTORY)
  EXECUTE_PROCESS(COMMAND ${Protobuf_PROTOC_EXECUTABLE} --proto_path=${proto_path} --proto_path=${CMAKE_SOURCE_DIR}/Databases/MaterialsDatabase/ --cpp_out=${proto_out_path} ${proto_native})
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

# generate version info
EXECUTE_PROCESS(COMMAND ${CMAKE_SOURCE_DIR}/Version/generate_build_version.sh)

ADD_LIBRARY(libmusen_static STATIC ${lib_musen_src})
SET_TARGET_PROPERTIES(libmusen_static PROPERTIES
  VERSION ${VERSION}
  SOVERSION ${SOVERSION}
  PREFIX ""
  OUTPUT_NAME "libmusen"
)

SET(INSTALL_PATH_ROOT ${CMAKE_INSTALL_PREFIX}/)

# TARGET_LINK_LIBRARIES(libmusen_shared ${Protobuf_LIBRARY} ${CUDA_LIBRARIES})
TARGET_LINK_LIBRARIES(libmusen_static ${Protobuf_LIBRARY} ${CUDA_LIBRARIES})

IF(BUILD_CLI)
  ADD_EXECUTABLE(CMUSEN ${proto_src} ${proto_files})
  TARGET_LINK_LIBRARIES(CMUSEN libmusen_static)
  
  # link filesystem or experimantal/filesystem for gcc
  IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    IF (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
      TARGET_LINK_LIBRARIES(CMUSEN -lstdc++fs)
    ENDIF()
  ENDIF()
  
  INSTALL(TARGETS CMUSEN DESTINATION ${INSTALL_PATH_ROOT})
ENDIF(BUILD_CLI)

# TARGET_COMPILE_DEFINITIONS(libmusen_shared PRIVATE INSTALL_DOCS_PATH="${INSTALL_DOCS_PATH}")
TARGET_COMPILE_DEFINITIONS(libmusen_static PRIVATE INSTALL_DOCS_PATH="${INSTALL_DOCS_PATH}")

INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Databases/" DESTINATION ${INSTALL_PATH_ROOT}/Databases)
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Examples/"  DESTINATION ${INSTALL_PATH_ROOT}/Examples)
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Licenses/"  DESTINATION ${INSTALL_PATH_ROOT}/Licenses)
INSTALL(DIRECTORY "${CMAKE_SOURCE_DIR}/Documentation/"             DESTINATION ${INSTALL_PATH_ROOT}/Documentation)
INSTALL(FILES     "${CMAKE_SOURCE_DIR}/LICENSE"                    DESTINATION ${INSTALL_PATH_ROOT})

# INSTALL(TARGETS libmusen_shared DESTINATION ${CMAKE_INSTALL_LIBDIR})
# INSTALL(TARGETS libmusen_static DESTINATION ${CMAKE_INSTALL_LIBDIR})

IF(BUILD_GUI)
  ADD_SUBDIRECTORY("${CMAKE_SOURCE_DIR}/MusenGUI")
ENDIF(BUILD_GUI)
