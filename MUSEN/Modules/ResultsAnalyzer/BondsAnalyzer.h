/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzer.h"

class CBondsAnalyzer :
	public CResultsAnalyzer
{
public:
	CBondsAnalyzer();

	bool Export() override;
};

