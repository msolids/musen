/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

/*
 * This header includes Windows.h safely, disabling some error-prone macros defined there.
 */

#pragma once

#ifdef NOMINMAX
#include <Windows.h>
#else
#define NOMINMAX
#include <Windows.h>
#undef NOMINMAX
#endif
#undef GetCurrentTime