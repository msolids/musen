/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "FileMerger.h"
#include "MUSENFileFunctions.h"

CFileMerger::CFileMerger(CSystemStructure* _pSystemStructure)
{
	m_dProgressPercent = 0;
	m_sProgressMessage = "";
	m_sErrorMessage = "";
	m_nCurrentStatus = ERunningStatus::IDLE;
	m_vListOfFiles.clear();
	m_sResultFile = "";
	m_dLastTimePoint = 0;
	m_pSystemStructure = _pSystemStructure;
	m_bLoadMergedFile = false;
}

CSystemStructure* CFileMerger::GetSystemStrcuture() const
{
	return m_pSystemStructure;
}

double CFileMerger::GetProgressPercent()
{
	return m_dProgressPercent;
}

std::string& CFileMerger::GetProgressMessage()
{
	return m_sProgressMessage;
}

std::string& CFileMerger::GetErrorMessage()
{
	return m_sErrorMessage;
}

ERunningStatus CFileMerger::GetCurrentStatus() const
{
	return m_nCurrentStatus;
}

void CFileMerger::SetCurrentStatus(const ERunningStatus& _nNewStatus)
{
	m_nCurrentStatus = _nNewStatus;
}

void CFileMerger::SetListOfFiles(std::vector<std::string> &_vListOfFiles)
{
	m_vListOfFiles = _vListOfFiles;
}

void CFileMerger::SetResultFile(std::string _sResultFile)
{
	m_sResultFile = _sResultFile;
}

void CFileMerger::SetFlagOfLoadingMergedFile(bool _bIsLoadingMergedFile)
{
	m_bLoadMergedFile = _bIsLoadingMergedFile;
}

void CFileMerger::Merge()
{
	m_sProgressMessage = "Merging started. Please wait...";
	m_sErrorMessage = "";

	for (const auto& file : m_vListOfFiles)
		if (CSystemStructure::IsOldFileVersion(file))
		{
			m_sErrorMessage = "File '" + file + "' is in old format. Convert it to a new format before merging.";
			return;
		}

	CSystemStructure* pSystemStructure = new CSystemStructure();		 // system structure for result
	CSystemStructure::ELoadFileResult status = pSystemStructure->LoadFromFile(m_vListOfFiles[0]);					 // load first file to the result system structure

	pSystemStructure->SaveToFile(m_sResultFile);						 // save result system structure to result file
	pSystemStructure->LoadFromFile(m_sResultFile);						 // load result file to the result system structure

	m_dLastTimePoint = pSystemStructure->GetAllTimePoints()[pSystemStructure->GetAllTimePoints().size() - 1]; // last time point in the first file

	size_t nNumberOfObjects = pSystemStructure->GetTotalObjectsCount();		   // total number of objects in the first file
	int nNumberOfRemovedObjects = GetNumberOfRemovedObjects(pSystemStructure); // number of objects which will be removed after data snapshot
	nNumberOfObjects = nNumberOfObjects - nNumberOfRemovedObjects;

	m_dProgressPercent = double((1 * 100) / (m_vListOfFiles.size()));
	for (auto i = 1; i < m_vListOfFiles.size(); i++)
	{
		if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;

		CSystemStructure* pTempSS = new CSystemStructure();			// temporary system structure for getting data from current file
		status = pTempSS->LoadFromFile(m_vListOfFiles[i]);			// load current file to temporary system structure

		if (pTempSS->GetTotalObjectsCount() == nNumberOfObjects && m_vListOfFiles[i]!= pSystemStructure->GetFileName())
		{
			std::vector<double> vTimePoints = pTempSS->GetAllTimePoints(); // get all time points from current file
			double dCurrentSavingTimeStep = 0;
			if (vTimePoints.size() >= 2)
				dCurrentSavingTimeStep = vTimePoints[1] - vTimePoints[0];  // calculate saving time step for current file
			else
				continue;

			double dCurrTime = m_dLastTimePoint;
			for (auto j = 1; j < vTimePoints.size(); j++)
			{
				if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;
				m_sProgressMessage = "In progress... Current file #" + std::to_string(i) + ", time point in result file " + std::to_string(dCurrTime) + "[s]";
				dCurrTime = dCurrTime + dCurrentSavingTimeStep;
				if (dCurrTime > m_dLastTimePoint + vTimePoints[vTimePoints.size() - 1])
					dCurrTime = m_dLastTimePoint + vTimePoints[vTimePoints.size() - 1];

				for (auto k = 0; k < nNumberOfObjects; k++)
				{
					CPhysicalObject* pObjectNew = pTempSS->GetObjectByIndex(k);
					CPhysicalObject* pObject = pSystemStructure->GetObjectByIndex(k);
					if (pObjectNew)
					{
						if (pObjectNew->IsActive(vTimePoints[j]))
						{
							switch (pObjectNew->GetObjectType())
							{
							case SPHERE:
							{
								pObject->SetCoordinates(dCurrTime, pObjectNew->GetCoordinates(vTimePoints[j]));
								pObject->SetVelocity(dCurrTime, pObjectNew->GetVelocity(vTimePoints[j]));
								pObject->SetAngleVelocity(dCurrTime, pObjectNew->GetAngleVelocity(vTimePoints[j]));
								pObject->SetForce(dCurrTime, pObjectNew->GetForce(vTimePoints[j]));
								pObject->SetOrientation(dCurrTime, pObjectNew->GetOrientation(vTimePoints[j]));
								break;
							}
							case SOLID_BOND:
							{
								CSolidBond* pSBond = static_cast<CSolidBond*>(pSystemStructure->GetObjectByIndex(k));
								CSolidBond* pSBondNew = static_cast<CSolidBond*>(pTempSS->GetObjectByIndex(k));
								pSBond->SetForce(dCurrTime, pSBondNew->GetForce(vTimePoints[j]));
								pSBond->SetTangentialOverlap(dCurrTime, pSBondNew->GetTangentialOverlap(vTimePoints[j]));
								pSBond->SetTotalTorque(dCurrTime, pSBondNew->GetTotalTorque(vTimePoints[j]));
								break;
							}
							case LIQUID_BOND:
							{
								CLiquidBond* pLBond = static_cast<CLiquidBond*>(pSystemStructure->GetObjectByIndex(k));
								CLiquidBond* pLBondNew = static_cast<CLiquidBond*>(pTempSS->GetObjectByIndex(k));
								pLBond->SetForce(dCurrTime, pLBondNew->GetForce(vTimePoints[j]));
								break;
							}
							case TRIANGULAR_WALL:
							{
								CTriangularWall* pWall = static_cast<CTriangularWall*>(pSystemStructure->GetObjectByIndex(k));
								CTriangularWall* pWallNew = static_cast<CTriangularWall*>(pTempSS->GetObjectByIndex(k));
								pWall->SetPlaneCoord(dCurrTime, pWallNew->GetCoordVertex1(vTimePoints[j]), pWallNew->GetCoordVertex2(vTimePoints[j]), pWallNew->GetCoordVertex3(vTimePoints[j]));
								pWall->SetForce(dCurrTime, pWallNew->GetForce(vTimePoints[j]));
								pWall->SetVelocity(dCurrTime, pWallNew->GetVelocity(vTimePoints[j]));
								break;
							}
							default: break;
							}
						}
						else
						{
							double dActivityStart, dActivityEnd;
							pObject->GetActivityTimeInterval(&dActivityStart, &dActivityEnd);
							if (dActivityEnd > dCurrTime)
								pObject->SetEndActivityTime(dCurrTime);
						}
					}
				}
			}
			if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;
			m_dLastTimePoint = dCurrTime;
		}
		else
		{
			if (m_vListOfFiles[i] != pSystemStructure->GetFileName())
				m_sErrorMessage = "File " + m_vListOfFiles[i] + " is already in use. Close file and repeat the merging.";
			else
				m_sErrorMessage = "Different number of objects in the file " + m_vListOfFiles[i] + " and in the previous file.";
			m_dProgressPercent = 0;
			delete pTempSS;
			break;
		}
		nNumberOfRemovedObjects = GetNumberOfRemovedObjects(pTempSS);
		delete pTempSS;
		nNumberOfObjects = nNumberOfObjects - static_cast<size_t>(nNumberOfRemovedObjects);
		m_dProgressPercent = double(((i + 1) * 100) / (m_vListOfFiles.size()));
	}
	pSystemStructure->SaveToFile(m_sResultFile);
	delete pSystemStructure;

	if (m_sErrorMessage == "")
	{
		if (m_bLoadMergedFile)
		{
			if (!m_pSystemStructure)
			{
				m_pSystemStructure = new CSystemStructure();
			}
			m_pSystemStructure->LoadFromFile(m_sResultFile);
		}
	}
	else
		MUSENFileFunctions::removeFile(m_sResultFile);

	if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED)
	{
		m_nCurrentStatus = ERunningStatus::IDLE;
	}
	m_sProgressMessage = "Merging finished";
}

int CFileMerger::GetNumberOfRemovedObjects(CSystemStructure* _pSystemStrcuture)
{
	std::vector<double> vAllTimePoints = _pSystemStrcuture->GetAllTimePoints();
	uint32_t nNumberOfRemovedObjects = 0;
	for (auto i = _pSystemStrcuture->GetTotalObjectsCount() - 1; i > 0; i--)
	{
		CPhysicalObject* pObject = _pSystemStrcuture->GetObjectByIndex(i);
		if ((pObject) && (pObject->IsActive(vAllTimePoints[vAllTimePoints.size() - 1])))
		{
			break;
		}
		nNumberOfRemovedObjects++;
	}
	return nNumberOfRemovedObjects;
}