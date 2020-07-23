/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"
#include "ScriptAnalyzer.h"
#include "ResultsAnalyzer.h"

class CConsoleResultsAnalyzer
{
	/// Implemented analyzer types.
	enum class EAnalyzerTypes
	{
		GeometryAnalyzer,
		BondsAnalyzer,
		ParticlesAnalyzer,
		AgglomeratesAnalyzer,
	};

	std::ostream& m_out;
	std::ostream& m_err;

	std::string m_sFileExt;			// Extension of resulting file.
	std::string m_sOutputPrefix;

	SJob m_job;

public:
	/// Custom error handling.
	enum class EProcessResultType
	{
		Success = 0,
		CantWriteFile = 1,
		WrongInput = 2,
		WrongNumberArguments = 3
	};

	CConsoleResultsAnalyzer(std::ostream& _out = std::cout, std::ostream& _err = std::cerr);

	EProcessResultType SetupAnalyzer(const std::vector<std::string>& _commandSet, CSystemStructure& _systemStructure, std::shared_ptr<CResultsAnalyzer>& _outAnalyzer) const;
	void EvaluateResults(const SJob& _job, CSystemStructure& _systemStructure);

private:
	std::vector<std::shared_ptr<CResultsAnalyzer>> GetAnalyzers(CSystemStructure& _systemStructure) const;

	EProcessResultType WriteError(EProcessResultType _errorType, const std::string& _comment = "") const;

	// Sub steps
	EProcessResultType SetProperties(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const;
	EProcessResultType SetResultType(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const;
	EProcessResultType SetGeometry(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const;
};