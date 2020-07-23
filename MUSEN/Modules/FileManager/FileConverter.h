/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"

class CFileConverter
{
private:
	std::string m_sProgressMessage;	// Progress description.
	std::string m_sErrorMessage;	// Error description.
	std::string m_sFileName;		// Name of input file.
	double m_dProgressPercent;		// Percent of progress.

public:
	CFileConverter(const std::string& _sFileName);

	// Returns current percent of converting.
	double GetProgressPercent() const;
	// Returns string with status description.
	std::string GetProgressMessage() const;
	// Returns string with error description.
	std::string GetErrorMessage() const;
	// Main function of converting.
	void ConvertFileToNewFormat();

private:
	void ConvertFileV0ToV2(CSystemStructure* _pSystemStructure);
	void ConvertFileV1ToV2(CSystemStructure* _pSystemStructure);
};