/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#define MACRO_TOSTRING(X) MACRO_TOSTRING_L2(X)
#define MACRO_TOSTRING_L2(X) #X

#define VERSION_0			1
#define VERSION_1			70
#define VERSION_2			0
#define MUSEN_VERSION		VERSION_0,VERSION_1,VERSION_2
#define MUSEN_VERSION_STR	MACRO_TOSTRING(VERSION_0) "." MACRO_TOSTRING(VERSION_1) "." MACRO_TOSTRING(VERSION_2)
