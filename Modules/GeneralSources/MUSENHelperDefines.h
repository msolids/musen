/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#define MACRO_CONCAT(X,Y) MACRO_CONCAT_L2(X,Y)
#define MACRO_CONCAT_L2(X,Y) X ## Y
#define MACRO_TOSTRING(X) MACRO_TOSTRING_L2(X)
#define MACRO_TOSTRING_L2(X) #X
