/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"
#include "ScriptAnalyzer.h"
#include "ResultsAnalyzer.h"
#include "SimulatorManager.h"

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

	// Set up analyzer as a monitor by adding it's evaluation into additionalSavingSteps of the Simulation Manager. Closing of the futput files happens if the CResultsAnalyzer is destructed (e.g. if the SimulationManger is destructed when the return value of this function is ignored). 
	std::vector<std::shared_ptr<CResultsAnalyzer>> SetupMonitor(const SJob& _job, CSystemStructure& _systemStructure, CSimulatorManager& _simManager);

private:
	std::vector<std::shared_ptr<CResultsAnalyzer>> GetAnalyzers(CSystemStructure& _systemStructure, std::vector<std::string> _vAnalzyerSettings) const;

	EProcessResultType WriteError(EProcessResultType _errorType, const std::string& _comment = "") const;

	// Sub steps
	EProcessResultType SetProperties(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const;
	EProcessResultType SetResultType(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const;
	EProcessResultType SetGeometry(const std::string& _inputString, const EAnalyzerTypes& _analyzerType, CResultsAnalyzer& _analyzer) const;
};