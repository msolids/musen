/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ScriptJob.h"

class CScriptAnalyzer
{
public:
	std::vector<SJob> m_jobs;

public:
	CScriptAnalyzer(const std::string& _fileName = "");

	bool AnalyzeFile(const std::string& _fileName);
	void ProcessLine(const std::string& _line, std::ostream& _out = std::cout);

	size_t JobsCount() const;
	std::vector<SJob> Jobs() const;
};
