/* Copyright (c) 2013-2020, MUSEN Development Team.
 * Copyright (c) 2024, DyssolTEC GmbH.
 * All rights reserved. This file is part of MUSEN framework http://msolids.net/musen.
 * See LICENSE file for license and warranty information. */

#include "FileMerger.h"
#include "MUSENFileFunctions.h"

CFileMerger::CFileMerger(CSystemStructure* _systemStructure)
	: m_systemStructure{ _systemStructure }
{
}

CSystemStructure* CFileMerger::GetSystemStructure() const
{
	return m_systemStructure;
}

double CFileMerger::GetProgress() const
{
	return m_progress;
}

std::string CFileMerger::GetProgressMessage() const
{
	return m_progressMessage;
}

std::string CFileMerger::GetErrorMessage() const
{
	return m_errorMessage;
}

ERunningStatus CFileMerger::GetCurrentStatus() const
{
	return m_status;
}

void CFileMerger::SetCurrentStatus(const ERunningStatus& _status)
{
	m_status = _status;
}

void CFileMerger::SetListOfFiles(const std::vector<std::string>& _listOfFiles)
{
	m_listOfFiles = _listOfFiles;
}

void CFileMerger::SetResultFile(const std::string& _resultFile)
{
	m_resultFile = _resultFile;
}

void CFileMerger::SetFlagOfLoadingMergedFile(bool _flag)
{
	m_loadMergedFile = _flag;
}

void CFileMerger::Merge()
{
	if (m_listOfFiles.empty())
		return;

	m_progressMessage = "Merging started. Please wait...";
	m_errorMessage.clear();

	for (const auto& file : m_listOfFiles)
		if (CSystemStructure::IsOldFileVersion(file))
		{
			m_errorMessage = "File '" + file + "' is in old format. Convert it to a new format before merging.";
			return;
		}

	auto* resultSS = new CSystemStructure{};		// system structure for result
	resultSS->LoadFromFile(m_listOfFiles.front());	// load first file to the result system structure

	resultSS->SaveToFile(m_resultFile);				// save result system structure to result file
	resultSS->LoadFromFile(m_resultFile);			// load result file to the result system structure

	double lastTimePoint = resultSS->GetAllTimePoints().back(); // last time point in the last processed file

	size_t numberOfObjects = resultSS->GetTotalObjectsCount();					// total number of objects in the file
	const size_t numberOfRemovedObjects = GetNumberOfRemovedObjects(resultSS);	// number of objects which will be removed after data snapshot
	numberOfObjects = numberOfObjects - numberOfRemovedObjects;

	m_progress = 100. / static_cast<double>(m_listOfFiles.size());
	for (size_t iFile = 1; iFile < m_listOfFiles.size(); ++iFile)
	{
		if (m_status == ERunningStatus::TO_BE_STOPPED) break;

		auto currentSS = std::make_unique<CSystemStructure>();	// temporary system structure for getting data from current file
		currentSS->LoadFromFile(m_listOfFiles[iFile]);			// load current file to temporary system structure

		if (currentSS->GetTotalObjectsCount() == numberOfObjects && m_listOfFiles[iFile] != resultSS->GetFileName())
		{
			const std::vector<double> timePoints = currentSS->GetAllTimePoints(); // get all the time points from current file
			if (timePoints.size() < 2)
				continue;

			double currentTime = lastTimePoint;
			for (size_t iTime = 1; iTime < timePoints.size(); ++iTime)
			{
				if (m_status == ERunningStatus::TO_BE_STOPPED)
					break;

				m_progressMessage = "Current file #" + std::to_string(iFile) + ", time point " + std::to_string(currentTime) + "s";
				const double timeStep = timePoints[iTime] - timePoints[iTime - 1];  // current saving time step for current file
				currentTime += timeStep;
				if (currentTime > lastTimePoint + timePoints.back())
					currentTime = lastTimePoint + timePoints.back();
				resultSS->PrepareTimePointForWrite(currentTime);
				currentSS->PrepareTimePointForRead(timePoints[iTime]);

				for (size_t k = 0; k < numberOfObjects; ++k)
				{
					const CPhysicalObject* objectNew = currentSS->GetObjectByIndex(k);
					if (!objectNew) continue;
					CPhysicalObject* object = resultSS->GetObjectByIndex(k);

					if (objectNew->IsActive(timePoints[iTime]))
					{
						switch (objectNew->GetObjectType())
						{
						case SPHERE:
						{
							object->SetCoordinates(objectNew->GetCoordinates());
							object->SetVelocity(objectNew->GetVelocity());
							object->SetAngleVelocity(objectNew->GetAngleVelocity());
							object->SetForce(objectNew->GetForce());
							object->SetOrientation(objectNew->GetOrientation());
							break;
						}
						case SOLID_BOND:
						{
							const auto* bond = dynamic_cast<CSolidBond*>(resultSS->GetObjectByIndex(k));
							const auto* bondNew = dynamic_cast<CSolidBond*>(currentSS->GetObjectByIndex(k));
							bond->SetForce(bondNew->GetForce());
							bond->SetTangentialOverlap(bondNew->GetTangentialOverlap());
							bond->SetTotalTorque(bondNew->GetTotalTorque());
							break;
						}
						case LIQUID_BOND:
						{
							const auto* bond = dynamic_cast<CLiquidBond*>(resultSS->GetObjectByIndex(k));
							const auto* bondNew = dynamic_cast<CLiquidBond*>(currentSS->GetObjectByIndex(k));
							bond->SetForce(bondNew->GetForce());
							break;
						}
						case TRIANGULAR_WALL:
						{
							const auto* wall = dynamic_cast<CTriangularWall*>(resultSS->GetObjectByIndex(k));
							const auto* wallNew = dynamic_cast<CTriangularWall*>(currentSS->GetObjectByIndex(k));
							wall->SetPlaneCoord(wallNew->GetCoordVertex1(), wallNew->GetCoordVertex2(), wallNew->GetCoordVertex3());
							wall->SetForce(wallNew->GetForce());
							wall->SetVelocity(wallNew->GetVelocity());
							break;
						}
						default: break;
						}
					}
					else
					{
						auto [activityStartOld, activityEndOld] = object->GetActivityTimeInterval();
						auto [activityStartNew, activityEndNew] = objectNew->GetActivityTimeInterval();

						if (activityEndOld == DEFAULT_ACTIVITY_END && activityEndNew != DEFAULT_ACTIVITY_END)
							object->SetEndActivityTime(lastTimePoint + activityEndNew);
					}
				}
			}

			if (m_status == ERunningStatus::TO_BE_STOPPED)
				break;

			lastTimePoint = currentTime;
		}
		else
		{
			if (m_listOfFiles[iFile] != resultSS->GetFileName())
				m_errorMessage = "File " + m_listOfFiles[iFile] + " is already in use. Close file and repeat the merging.";
			else
				m_errorMessage = "Different number of objects in the file " + m_listOfFiles[iFile] + " and in the previous file.";
			m_progress = 0;
			break;
		}
		numberOfObjects -= GetNumberOfRemovedObjects(currentSS.get());
		m_progress = static_cast<double>(iFile + 1) * 100. / static_cast<double>(m_listOfFiles.size());
	}
	resultSS->SaveToFile(m_resultFile);
	delete resultSS;

	if (m_errorMessage.empty())
	{
		if (m_loadMergedFile)
		{
			if (!m_systemStructure)
				m_systemStructure = new CSystemStructure{};
			m_systemStructure->LoadFromFile(m_resultFile);
		}
	}
	else
		MUSENFileFunctions::removeFile(m_resultFile);

	if (m_status == ERunningStatus::TO_BE_STOPPED)
		m_status = ERunningStatus::IDLE;
	m_progressMessage = "Merging finished";
}

size_t CFileMerger::GetNumberOfRemovedObjects(const CSystemStructure* _systemStructure)
{
	const std::vector<double> timePoints = _systemStructure->GetAllTimePoints();
	size_t numberOfRemovedObjects = 0;
	for (auto i = _systemStructure->GetTotalObjectsCount() - 1; i > 0; --i)
	{
		const CPhysicalObject* object = _systemStructure->GetObjectByIndex(i);
		if (object && object->IsActive(timePoints.back()))
		{
			break;
		}
		numberOfRemovedObjects++;
	}
	return numberOfRemovedObjects;
}
