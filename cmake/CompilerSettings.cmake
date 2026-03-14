# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen. 
# See LICENSE file for license and warranty information.

# Compiler and linker settings for MSVC and GCC.

if(MSVC)
  # --- Compile flags (all configurations) ---
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/MP>        # Multi-processor compilation
    $<$<COMPILE_LANGUAGE:C,CXX>:/W3>        # Warning level 3
    $<$<COMPILE_LANGUAGE:C,CXX>:/w34062>    # Warn on unhandled enum values in switch
  )
  # Suppress warnings from external/system headers
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.29")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:C,CXX>:/external:anglebrackets>
      $<$<COMPILE_LANGUAGE:C,CXX>:/external:W0>
    )
  endif()
  add_compile_definitions(
    UNICODE
    _UNICODE
    _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
  )

  # --- Debug configuration ---
  set(CMAKE_C_FLAGS_DEBUG            "${CMAKE_C_FLAGS_DEBUG} /RTC1 /JMC /sdl")
  set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} /RTC1 /JMC /sdl")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG    " /INCREMENTAL")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_DEBUG " /INCREMENTAL")

  # --- Release configuration ---
  set(CMAKE_C_FLAGS_RELEASE            "${CMAKE_C_FLAGS_RELEASE} /Oi /Ot /GL /Gy /Oy /GS-")
  set(CMAKE_CXX_FLAGS_RELEASE          "${CMAKE_CXX_FLAGS_RELEASE} /Oi /Ot /GL /Gy /Oy /GS-")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE    " /OPT:REF /OPT:ICF /LTCG")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " /OPT:REF /OPT:ICF /LTCG")
  string(APPEND CMAKE_STATIC_LINKER_FLAGS_RELEASE " /LTCG")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # --- Compile flags (all configurations) ---
  add_compile_options(-Wall -Wshadow -Wno-unused-parameter -fstack-protector-strong)

  # --- Release configuration ---
  add_compile_definitions($<$<CONFIG:Release>:_FORTIFY_SOURCE=2>)
  string(APPEND CMAKE_C_FLAGS_RELEASE   " -flto")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE    " -flto")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " -flto")
endif()
