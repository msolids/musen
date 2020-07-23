/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"

class CFileMerger
{
public:
	CFileMerger(CSystemStructure* _pSystemStructure);

	// Returns current percent of merging
	double GetProgressPercent();
	// Returns string with status description
	std::string& GetProgressMessage();
	// Returns string with erros description
	std::string& GetErrorMessage();
	// Returns status of merging files function (IDLE, RUNNING, etc.)
	ERunningStatus GetCurrentStatus() const;
	// Returns poinbter to system structure
	CSystemStructure* GetSystemStrcuture() const;
	// Sets status of merging files function (IDLE, RUNNING, etc.)
	void SetCurrentStatus(const ERunningStatus& _nNewStatus);
	// Sets list of files which should be merged
	void SetListOfFiles(std::vector<std::string> &_vListOfFiles);
	// Sets path to the result file
	void SetResultFile(std::string _sResultFile);
	// Sets flag of loading result file after merging
	void SetFlagOfLoadingMergedFile(bool _bIsLoadingMergedFile);
	// Main function: Merge files into one output file
	void Merge();

private:
	CSystemStructure * m_pSystemStructure;	// pointer to system structure

	double m_dProgressPercent;				// percent of progress (is used for progress bar)
	std::string m_sProgressMessage;			// progress description
	std::string m_sErrorMessage;			// error description
	ERunningStatus m_nCurrentStatus;		// current status of the generator: IDLE, RUNNING, etc.

	double m_dLastTimePoint;				// last time point in the result file
	bool m_bLoadMergedFile;					// flag of loading result file after merging
	std::string m_sResultFile;				// path to the result file

	std::vector<std::string> m_vListOfFiles; // list of paths to files which should be merged

	// Returns number of objects which will be removed from system structure after making a snapshot
	int GetNumberOfRemovedObjects(CSystemStructure* _pSystemStrcuture);
};

