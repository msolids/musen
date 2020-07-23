/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"

class CClearSpecificTimePoints
{
	CSystemStructure* m_pSystemStructure;	// pointer to system structure

	double m_dProgressPercent;				// percent of progress (is used for progress bar)
	std::string m_sProgressMessage;			// progress description
	std::string m_sErrorMessage;			// error description

	std::vector<double> m_vAllTimePoints;			// all time points
	std::vector<size_t> m_vIndexesOfSelectedTPs;	// indexes of selected time points

public:
	CClearSpecificTimePoints(CSystemStructure* _pSystemStructure);

	// Returns current percent of merging
	double GetProgressPercent();
	// Returns string with status description
	std::string& GetProgressMessage();
	// Returns string with erros description
	std::string& GetErrorMessage();

	// Sets indexes of selected time points
	void SetIndexesOfSelectedTPs(std::vector<size_t>& _vIndexesOfSelectedTPs);
	// Main function of removing
	void Remove();
};

