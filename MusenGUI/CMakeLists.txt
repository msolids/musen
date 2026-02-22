# Copyright (c) 2023, MUSEN Development Team. 
# Copyright (c) 2025, DyssolTEC GmbH. 
# All rights reserved. This file is part of MUSEN framework. See LICENSE file for license and warranty information.

SET(INSTALL_STYLES_PATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/MUSEN/Styles)

SET(lib_musen_gui_names
  "MusenGUI"
  "QTDialogs"
)

# gather all sources
FOREACH(folder_name ${lib_musen_gui_names})
  FILE(GLOB_RECURSE src ${CMAKE_SOURCE_DIR}/${folder_name}/*.cpp ${CMAKE_SOURCE_DIR}/${folder_name}/*.rc ${CMAKE_SOURCE_DIR}/${folder_name}/*.ui ${CMAKE_SOURCE_DIR}/${folder_name}/*.qrc)
  FILE(GLOB_RECURSE hdr ${CMAKE_SOURCE_DIR}/${folder_name}/*.h)
  SET(lib_musen_gui_src ${lib_musen_gui_src} ${src})
  SET(lib_musen_gui_hdr ${lib_musen_gui_hdr} ${hdr})
ENDFOREACH(folder_name ${lib_musen_gui_names})

# gather include directories
FOREACH(hdr_file ${lib_musen_gui_hdr})
  GET_FILENAME_COMPONENT(hdr_path ${hdr_file} PATH)
  SET(hdr_paths ${hdr_paths} ${hdr_path})
ENDFOREACH()
LIST(REMOVE_DUPLICATES hdr_paths)
INCLUDE_DIRECTORIES(${hdr_paths})

find_package(Qt6 QUIET COMPONENTS OpenGL OpenGLWidgets Widgets)
if (NOT Qt6_FOUND)
    find_package(Qt5 REQUIRED COMPONENTS OpenGL Widgets)
endif()
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTORCC ON)
SET(CMAKE_AUTOUIC ON)

SET(OpenGL_GL_PREFERENCE GLVND)
FIND_PACKAGE(OpenGL REQUIRED)
INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIRS})

ADD_EXECUTABLE(MUSEN ${lib_musen_gui_src})

if (Qt6_FOUND)
  TARGET_LINK_LIBRARIES(MUSEN libmusen_static Qt::OpenGL Qt::OpenGLWidgets Qt::Widgets ${OPENGL_LIBRARIES})
else()
  TARGET_LINK_LIBRARIES(MUSEN libmusen_static Qt5::OpenGL Qt5::Widgets ${OPENGL_LIBRARIES})
endif()

# link filesystem or experimantal/filesystem for gcc
IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  IF (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    TARGET_LINK_LIBRARIES(MUSEN -lstdc++fs)
  ENDIF()
ENDIF()

TARGET_COMPILE_DEFINITIONS(MUSEN PRIVATE INSTALL_DOCS_PATH="${INSTALL_DOCS_PATH}")

INSTALL(TARGETS MUSEN DESTINATION ${INSTALL_PATH_ROOT})
IF(INSTALL_AUX_DATA)
  INSTALL(FILES "${CMAKE_SOURCE_DIR}/Installers/Data/MUSEN.ini"  DESTINATION ${INSTALL_PATH_ROOT})
ENDIF(INSTALL_AUX_DATA)
