/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

struct SOptionalVariables
{
	bool bThermals = false;
	bool bTangentialPlasticity = false;

	SOptionalVariables& operator|=(const SOptionalVariables &b)
	{
		bThermals = bThermals || b.bThermals;
		bTangentialPlasticity = bTangentialPlasticity || b.bTangentialPlasticity;
		return *this;
	};
};
