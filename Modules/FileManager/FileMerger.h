/* Copyright (c) 2013-2020, MUSEN Development Team.
 * Copyright (c) 2024, DyssolTEC GmbH.
 * All rights reserved. This file is part of MUSEN framework http://msolids.net/musen.
 * See LICENSE file for license and warranty information. */

#pragma once

#include "SystemStructure.h"

class CFileMerger
{
public:
	explicit CFileMerger(CSystemStructure* _systemStructure);

	/// Returns pointer to system structure.
	[[nodiscard]] CSystemStructure* GetSystemStructure() const;
	/// Returns current progress of merging in percent.
	[[nodiscard]] double GetProgress() const;
	/// Returns string with status description.
	[[nodiscard]] std::string GetProgressMessage() const;
	/// Returns string with error description.
	[[nodiscard]] std::string GetErrorMessage() const;
	/// Returns current status of merging (IDLE, RUNNING, etc.).
	[[nodiscard]] ERunningStatus GetCurrentStatus() const;

	/// Sets current status of merging (IDLE, RUNNING, etc.).
	void SetCurrentStatus(const ERunningStatus& _status);
	/// Sets list of files which should be merged.
	void SetListOfFiles(const std::vector<std::string>& _listOfFiles);
	/// Sets path to the result file.
	void SetResultFile(const std::string& _resultFile);
	/// Sets flag of loading result file after merging.
	void SetFlagOfLoadingMergedFile(bool _flag);

	/// Main function: Merge files into one output file.
	void Merge();

private:
	/// Returns the number of objects which will be removed from the system structure after making a snapshot.
	static size_t GetNumberOfRemovedObjects(const CSystemStructure* _systemStructure);

private:
	CSystemStructure* m_systemStructure{};	///< Pointer to system structure.

	double m_progress{};								///< Percentage of merging progress.
	std::string m_progressMessage{};					///< Description of the current step.
	std::string m_errorMessage{};						///< Error description.
	ERunningStatus m_status{ ERunningStatus::IDLE };	///< Current status of the merger: IDLE, RUNNING, etc.

	bool m_loadMergedFile{ false };	///< Whether to load result file after merging.
	std::string m_resultFile{};		///< Path to the result file.

	std::vector<std::string> m_listOfFiles; ///< List of paths to files which should be merged.
};

