# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Find/build external dependencies: zlib, protobuf, Qt, OpenGL.

include(FetchContent)
# Hide FetchContent internals from the CMake GUI
set(FETCHCONTENT_QUIET ON CACHE INTERNAL "")
mark_as_advanced(
  FETCHCONTENT_BASE_DIR
  FETCHCONTENT_FULLY_DISCONNECTED
  FETCHCONTENT_UPDATES_DISCONNECTED
)

# ============================================================================
# zlib (needed by protobuf)
# ============================================================================
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  set(ZLIB_USE_STATIC_LIBS ON)
endif()
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  message(STATUS "System zlib not found, building from source...")
  FetchContent_Declare(zlib
    URL      https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz
    URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
  )
  set(ZLIB_BUILD_TESTING OFF CACHE INTERNAL "")
  set(ZLIB_BUILD_SHARED  OFF CACHE INTERNAL "")
  set(ZLIB_BUILD_STATIC  ON  CACHE INTERNAL "")
  set(ZLIB_INSTALL       OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(zlib)
  # Hide zlib's internal options from the CMake GUI
  mark_as_advanced(FETCHCONTENT_SOURCE_DIR_ZLIB FETCHCONTENT_UPDATES_DISCONNECTED_ZLIB)
  foreach(_var
      ZLIB_BUILD_ADA ZLIB_BUILD_BLAST ZLIB_BUILD_IOSTREAM3 ZLIB_BUILD_MINIZIP
      ZLIB_BUILD_PUFF ZLIB_BUILD_TESTZLIB ZLIB_BUILD_ZLIB1_DLL
      ZLIB_WITH_CRC32VX ZLIB_WITH_INFBACK9 ZLIB_WITH_GVMAT64)
    if(DEFINED CACHE{${_var}})
      set(${_var} "${${_var}}" CACHE INTERNAL "")
    endif()
  endforeach()
  # ZLIB::ZLIB alias is only created for the shared target.
  # Provide it for the static target so consumers (protobuf, MUSEN) can use the standard name.
  if(NOT TARGET ZLIB::ZLIB AND TARGET ZLIB::ZLIBSTATIC)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  endif()
endif()

# ============================================================================
# protobuf
# ============================================================================
set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf QUIET)
if(NOT Protobuf_FOUND)
  message(STATUS "System protobuf not found, building from source...")
  FetchContent_Declare(protobuf
    URL        https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-cpp-3.21.12.tar.gz
    URL_HASH   SHA256=4eab9b524aa5913c6fffb20b2a8abf5ef7f95a80bc0701f3a6dbb4c607f73460
    SOURCE_SUBDIR cmake
  )
  set(protobuf_BUILD_TESTS    OFF CACHE INTERNAL "")
  set(protobuf_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(protobuf_WITH_ZLIB      ON  CACHE INTERNAL "")
  set(protobuf_INSTALL        OFF CACHE INTERNAL "")
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE INTERNAL "")
  # Suppress warnings from protobuf's outdated CMakeLists.txt
  set(_dep_save_warn_deprecated ${CMAKE_WARN_DEPRECATED})
  set(_dep_save_message_level "${CMAKE_MESSAGE_LOG_LEVEL}")
  set(CMAKE_WARN_DEPRECATED OFF CACHE INTERNAL "")
  set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
  FetchContent_MakeAvailable(protobuf)
  # Hide protobuf's internal options from the CMake GUI
  mark_as_advanced(FETCHCONTENT_SOURCE_DIR_PROTOBUF FETCHCONTENT_UPDATES_DISCONNECTED_PROTOBUF)
  foreach(_var protobuf_BUILD_CONFORMANCE protobuf_BUILD_LIBPROTOC protobuf_BUILD_PROTOC_BINARIES
               protobuf_BUILD_SHARED_LIBS protobuf_DISABLE_RTTI protobuf_MSVC_STATIC_RUNTIME
               Protobuf_SRC_ROOT_FOLDER)
    if(DEFINED CACHE{${_var}})
      set(${_var} "${${_var}}" CACHE INTERNAL "")
    endif()
  endforeach()
  # Restore suppressed warnings
  set(CMAKE_MESSAGE_LOG_LEVEL "${_dep_save_message_level}")
  if(DEFINED _dep_save_warn_deprecated)
    set(CMAKE_WARN_DEPRECATED ${_dep_save_warn_deprecated} CACHE INTERNAL "")
  else()
    unset(CMAKE_WARN_DEPRECATED CACHE)
  endif()
  # Ensure protoc target is available
  if(NOT TARGET protobuf::protoc)
    if(TARGET protoc)
      add_executable(protobuf::protoc ALIAS protoc)
    endif()
  endif()
endif()

# ============================================================================
# Qt and OpenGL (needed by MUSEN)
# ============================================================================
if(MUSEN_BUILD_GUI)
  # On Windows, hint at the common Qt installation directory
  if(WIN32 AND NOT Qt6_DIR)
    file(GLOB _qt_hints "C:/Qt/*/msvc*_64")
    if(_qt_hints)
      list(SORT _qt_hints COMPARE NATURAL ORDER DESCENDING)
      list(APPEND CMAKE_PREFIX_PATH ${_qt_hints})
    endif()
  endif()
  find_package(Qt6 QUIET COMPONENTS OpenGL OpenGLWidgets Widgets)
  if(NOT Qt6_FOUND)
    find_package(Qt5 REQUIRED COMPONENTS OpenGL Widgets)
  endif()
  # Hide Qt's internal options from the CMake GUI
  mark_as_advanced(
    Qt5Core_DIR Qt5Gui_DIR Qt5OpenGL_DIR Qt5Widgets_DIR
    Qt6Core_DIR Qt6CoreTools_DIR Qt6EntryPointPrivate_DIR
    Qt6Gui_DIR Qt6GuiTools_DIR Qt6OpenGL_DIR Qt6OpenGLWidgets_DIR
    Qt6Widgets_DIR Qt6WidgetsTools_DIR
    QT_ADDITIONAL_HOST_PACKAGES_PREFIX_PATH QT_ADDITIONAL_PACKAGES_PREFIX_PATH
    WINDEPLOYQT_EXECUTABLE
  )
  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  # Detect Qt bin directory (used for VS debugger PATH and installer)
  if(Qt6_FOUND)
    get_target_property(_qt_core_loc Qt6::Core IMPORTED_LOCATION_RELEASE)
    if(NOT _qt_core_loc)
      get_target_property(_qt_core_loc Qt6::Core IMPORTED_LOCATION)
    endif()
  else()
    get_target_property(_qt_core_loc Qt5::Core IMPORTED_LOCATION_RELEASE)
    if(NOT _qt_core_loc)
      get_target_property(_qt_core_loc Qt5::Core IMPORTED_LOCATION)
    endif()
  endif()
  if(_qt_core_loc)
    get_filename_component(MUSEN_QT_BIN_DIR "${_qt_core_loc}" DIRECTORY)
    get_filename_component(MUSEN_QT_PREFIX "${MUSEN_QT_BIN_DIR}" DIRECTORY)
  endif()
endif()
