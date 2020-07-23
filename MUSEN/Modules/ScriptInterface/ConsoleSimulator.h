/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SimulatorManager.h"
#include "ScriptJob.h"

class CConsoleSimulator
{
	std::ostream& m_out;
	std::ostream& m_err;

	SJob m_job;

	CSystemStructure& m_systemStructure;

	CAgglomeratesDatabase m_agglomeratesDatabase;
	CGenerationManager m_generationManager;
	CModelManager m_modelManager;
	CSimulatorManager m_simulatorManager;

public:
	CConsoleSimulator(CSystemStructure& _systemStructure, std::ostream& _out = std::cout, std::ostream& _err = std::cerr);

	CSimulatorManager& GetSimulatorManager(); // Returns a reference to the current simulator manager.

	void Simulate(const SJob* _job = nullptr);

	void Initialize(const SJob* _job = nullptr); // Apply all settings from job.

private:
	void SetupSystemStructure() const;	// Apply settings of m_job to m_systemStructure.
	void SetupGenerationManager();		// Apply settings of m_job to m_generationManager.
	void SetupModelManager();			// Apply settings of m_job to m_modelManager.
	void SetupSimulationManager();		// Apply settings of m_job to m_simulatorManager.

	// Checks all required settings and prints found warnings and errors, returns true if the  simulation can be started.
	bool SimulationPrecheck() const;
	// Runs the simulation.
	void RunSimulation() const;

	// Prints information about simulation settings.
	void PrintSimulationInfo();
	// Prints information about chosen models and their settings.
	void PrintModelsInfo();
	// Prints information about GPU.
	void PrintGPUInfo() const;
	// Prints message _m.
	template<typename... Args>
	void PrintFormatted(const std::string& _message, Args... args) const;
};


