/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ScriptJob.h"

class CScriptRunner
{
	friend class CMMusen;

public:
	std::ostream m_out;
	std::ostream m_err;

	SJob m_job;
	CSystemStructure m_systemStructure;

public:
	CScriptRunner();

	// Main function - executes the _job.
	void RunScriptJob(const SJob& _job);

private:
	void PerformSimulation();	// Simulates the scene.
	void GeneratePackage();		// Generates package of particles.
	void GenerateBonds();		// Generates bonds.
	void AnalyzeResults();		// Performs results analysis.
	void GenerateSnapshot();	// Takes a snapshot.
	void ExportToText();		// Saves the scene as a text file.
	void ImportFromText();		// Loads the scene from a text file.
	void RunTests();			// Runs tests.
	void CompareFiles();		// Compares two files.

	// Loads the source file into m_systemStructure and saves it into result file.
	bool LoadAndResaveSystemStructure();
	// Loads the source file into m_systemStructure.
	bool LoadSourceFile();
	// Loads a file named _sourceFileName into _systemStructure.
	bool LoadMusenFile(const std::string& _sourceFileName, CSystemStructure& _systemStructure);
	// Apply settings of m_job to materials database.
	void ApplyMaterialParameters();
	// Prints out the results of file loading.
	void PrintFileLoadingInfo(const CSystemStructure::ELoadFileResult& _status);
};