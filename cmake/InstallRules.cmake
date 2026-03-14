# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Platform-specific install rules.

if(WIN32)
  # Windows: flat layout - executables, Qt DLLs, and data together
  set(_BIN_DIR ".")
  set(_DATA_DIR ".")
  set(_DOC_DIR "Documentation")
else()
  # Linux: FHS-compliant layout
  set(_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
  set(_DATA_DIR "${CMAKE_INSTALL_DATADIR}/musen")
  set(_DOC_DIR "${CMAKE_INSTALL_DATADIR}/doc/musen")
endif()

# --- Executables ---
if(MUSEN_BUILD_CLI)
  install(TARGETS CMUSEN RUNTIME DESTINATION "${_BIN_DIR}")
endif()
if(MUSEN_BUILD_GUI)
  install(TARGETS MUSEN RUNTIME DESTINATION "${_BIN_DIR}")

  # Qt runtime libraries (Windows only - on Linux, Qt is a system dependency)
  if(WIN32 AND MUSEN_QT_BIN_DIR)
    # Core DLLs
    foreach(_lib Qt6Core Qt6Gui Qt6OpenGL Qt6Widgets Qt6OpenGLWidgets)
      install(FILES "${MUSEN_QT_BIN_DIR}/${_lib}.dll" DESTINATION "${_BIN_DIR}" OPTIONAL)
    endforeach()
    # Platform plugin
    get_filename_component(_qt_plugins "${MUSEN_QT_BIN_DIR}/../plugins" ABSOLUTE)
    install(FILES "${_qt_plugins}/platforms/qwindows.dll" DESTINATION "${_BIN_DIR}/platforms" OPTIONAL)
    # Image format plugin
    install(FILES "${_qt_plugins}/imageformats/qjpeg.dll" DESTINATION "${_BIN_DIR}/imageformats" OPTIONAL)
  endif()

  # Style file
  install(FILES "${CMAKE_SOURCE_DIR}/MusenGUI/styles/musen_style1.qss" DESTINATION "${_BIN_DIR}/styles")
endif()

# --- Data files ---
if(MUSEN_INSTALL_DATA)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Databases/" DESTINATION "${_DATA_DIR}/Databases")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Examples/"  DESTINATION "${_DATA_DIR}/Examples")
  install(FILES     "${CMAKE_SOURCE_DIR}/LICENSE"                    DESTINATION "${_DATA_DIR}")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/Documentation/"             DESTINATION "${_DOC_DIR}")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/Installers/Data/Licenses/"  DESTINATION "${_DATA_DIR}/Licenses")
  if(WIN32)
    install(FILES "${CMAKE_SOURCE_DIR}/Installers/Data/MUSEN.ini" DESTINATION "${_DATA_DIR}")
  endif()
endif()