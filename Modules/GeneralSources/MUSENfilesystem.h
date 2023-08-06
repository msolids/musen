/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

/*
 * Includes <filesystem> or <experimental/filesystem> depending on compiler
 */

#pragma once

#ifndef USE_EXPERIMENTAL                              // not yet checked
	#if defined(__cpp_lib_filesystem)                 // feature test for filesystem
		#define USE_EXPERIMENTAL 0
	#elif defined(__cpp_lib_experimental_filesystem)  // feature test experimental
		#define USE_EXPERIMENTAL 1
	#elif !defined(__has_include)                     // no headers check available - assume experimental
		#define USE_EXPERIMENTAL 1
	#elif __has_include(<filesystem>)                 // include test for filesystem
		#ifdef _MSC_VER                               // for MSVC compiler
			#if __has_include(<yvals_core.h>)         // check and include header that defines "_HAS_CXX17"
				#include <yvals_core.h>
				#if defined(_HAS_CXX17) && _HAS_CXX17 // c++17 supported - assume filesystem
					#define USE_EXPERIMENTAL 0
				#endif
			#endif
			#ifndef USE_EXPERIMENTAL                  // filesystem not yet found - use experimental
				#define USE_EXPERIMENTAL 1
			#endif
		#else                                         // not MSVC - use filesystem
			#define USE_EXPERIMENTAL 0
		#endif
	#elif __has_include(<experimental/filesystem>)    // include test for experimental
		#define USE_EXPERIMENTAL 1
	#else                                             // no header found - error message
		#error No <filesystem> or <experimental/filesystem> found
	#endif
	#if USE_EXPERIMENTAL                              // found experimental
		#include <experimental/filesystem>
		namespace std                                 // add an alias for std::experimental::filesystem to std::filesystem
		{
			namespace filesystem = experimental::filesystem;
		}
	#else                                             // found filesystem
		#include <filesystem>
	#endif
#endif
