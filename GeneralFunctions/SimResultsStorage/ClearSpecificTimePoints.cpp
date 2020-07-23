/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ClearSpecificTimePoints.h"

CClearSpecificTimePoints::CClearSpecificTimePoints(CSystemStructure* _pSystemStructure)
{
	m_pSystemStructure = _pSystemStructure;
	m_vIndexesOfSelectedTPs.clear();
	m_vAllTimePoints.clear();
	m_dProgressPercent = 0;
	m_sProgressMessage = "";
	m_sErrorMessage = "";
}

double CClearSpecificTimePoints::GetProgressPercent()
{
	return m_dProgressPercent;
}

std::string& CClearSpecificTimePoints::GetProgressMessage()
{
	return m_sProgressMessage;
}

std::string& CClearSpecificTimePoints::GetErrorMessage()
{
	return m_sErrorMessage;
}

void CClearSpecificTimePoints::SetIndexesOfSelectedTPs(std::vector<size_t>& _vIndexesOfSelectedTPs)
{
	m_vIndexesOfSelectedTPs.clear();
	m_vIndexesOfSelectedTPs.reserve(_vIndexesOfSelectedTPs.size());
	m_vIndexesOfSelectedTPs = _vIndexesOfSelectedTPs;
}

void CClearSpecificTimePoints::Remove()
{
	m_sProgressMessage = "Removing started. Please wait...";
	m_sErrorMessage = "";

	// original file name
	std::string sOrigFileName = m_pSystemStructure->GetFileName();

	// make a copy of original mdem file
	m_sProgressMessage = "Make a copy of the original file...";
	std::string sTempFileName = sOrigFileName + "_copy";
	m_pSystemStructure->SaveToFile(sTempFileName);

	// clear all data after 0 time point in the original file
	m_sProgressMessage = "Load and clear the original file...";
	m_pSystemStructure->LoadFromFile(sOrigFileName);
	m_pSystemStructure->ClearAllStatesFrom(0);

	// create temp system structure from copy of original file
	m_sProgressMessage = "Load copied file...";
	CSystemStructure* pTempSystemStructure = new CSystemStructure();
	pTempSystemStructure->LoadFromFile(sTempFileName);

	m_sProgressMessage = "Preparing...";
	// get number of all objects
	size_t nTotalObjects = pTempSystemStructure->GetTotalObjectsCount();

	// get all time points
	std::vector<double> m_vAllTimePoints = pTempSystemStructure->GetAllTimePoints();
	if (m_vAllTimePoints.size() == 0)
		m_vAllTimePoints = pTempSystemStructure->GetAllTimePointsOldFormat();
	size_t nTotalTimePoints = m_vAllTimePoints.size();

	for (size_t i = 0; i < nTotalTimePoints; i++)
	{
		if (std::find(m_vIndexesOfSelectedTPs.begin(), m_vIndexesOfSelectedTPs.end(), i) != m_vIndexesOfSelectedTPs.end())
			continue; // skip time point which has to be removed

		double dCurrTime = m_vAllTimePoints[i];
		m_sProgressMessage = "In progress... Current time point " + std::to_string(dCurrTime) + " [s]";

		for (auto k = 0; k < nTotalObjects; k++)
		{
			CPhysicalObject* pObjectTemp = pTempSystemStructure->GetObjectByIndex(k);
			CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(k);
			if (pObjectTemp && pObject)
			{
				if (pObjectTemp->IsActive(dCurrTime))
				{
					switch (pObject->GetObjectType())
					{
					case SPHERE:
					{
						pObject->SetCoordinates(dCurrTime, pObjectTemp->GetCoordinates(dCurrTime));
						pObject->SetVelocity(dCurrTime, pObjectTemp->GetVelocity(dCurrTime));
						pObject->SetAngleVelocity(dCurrTime, pObjectTemp->GetAngleVelocity(dCurrTime));
						pObject->SetForce(dCurrTime, pObjectTemp->GetForce(dCurrTime));
						pObject->SetOrientation(dCurrTime, pObjectTemp->GetOrientation(dCurrTime));
						break;
					}
					case SOLID_BOND:
					{
						CSolidBond* pSBond = static_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex(k));
						CSolidBond* pSBondTemp = static_cast<CSolidBond*>(pTempSystemStructure->GetObjectByIndex(k));
						pSBond->SetForce(dCurrTime, pSBondTemp->GetForce(dCurrTime));
						pSBond->SetTangentialOverlap(dCurrTime, pSBondTemp->GetTangentialOverlap(dCurrTime));
						pSBond->SetTotalTorque(dCurrTime, pSBondTemp->GetTotalTorque(dCurrTime));
						break;
					}
					case LIQUID_BOND:
					{
						CLiquidBond* pLBond = static_cast<CLiquidBond*>(m_pSystemStructure->GetObjectByIndex(k));
						CLiquidBond* pLBondTemp = static_cast<CLiquidBond*>(pTempSystemStructure->GetObjectByIndex(k));
						pLBond->SetForce(dCurrTime, pLBondTemp->GetForce(dCurrTime));
						break;
					}
					case TRIANGULAR_WALL:
					{
						CTriangularWall* pWall = static_cast<CTriangularWall*>(m_pSystemStructure->GetObjectByIndex(k));
						CTriangularWall* pWallTemp = static_cast<CTriangularWall*>(pTempSystemStructure->GetObjectByIndex(k));
						pWall->SetPlaneCoord(dCurrTime, pWallTemp->GetCoordVertex1(dCurrTime), pWallTemp->GetCoordVertex2(dCurrTime), pWallTemp->GetCoordVertex3(dCurrTime));
						pWall->SetForce(dCurrTime, pWallTemp->GetForce(dCurrTime));
						pWall->SetVelocity(dCurrTime, pWallTemp->GetVelocity(dCurrTime));
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
			m_dProgressPercent = double(i *  nTotalObjects + k) * 100 / (nTotalTimePoints * nTotalObjects);
		}
	}

	m_sProgressMessage = "Save the final file...";
	m_pSystemStructure->SaveToFile(sOrigFileName);

	// remove temporary file
	delete pTempSystemStructure;
	MUSENFileFunctions::removeFile(sTempFileName);

	m_sProgressMessage = "Removing finished";
	m_sProgressMessage = 100;
}