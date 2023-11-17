/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "FileConverter.h"
#include "MUSENFileFunctions.h"

CFileConverter::CFileConverter(const std::string& _sFileName)
{
	m_sFileName = _sFileName;
	m_sProgressMessage = "";
	m_sErrorMessage = "";
	m_dProgressPercent = 0;
}

double CFileConverter::GetProgressPercent() const
{
	return m_dProgressPercent;
}

std::string CFileConverter::GetProgressMessage() const
{
	return m_sProgressMessage;
}

std::string CFileConverter::GetErrorMessage() const
{
	return m_sErrorMessage;
}

void CFileConverter::ConvertFileToNewFormat()
{
	m_sProgressMessage = "Transformation started. Please wait...";

	// load input file
	auto* pSystemStructure = new CSystemStructure();
	pSystemStructure->LoadFromFile(m_sFileName);

	if (pSystemStructure->FileVersion() == 0)
		ConvertFileV0ToV2(pSystemStructure);
	else
		ConvertFileV1ToV2(pSystemStructure);
}

void CFileConverter::ConvertFileV0ToV2(CSystemStructure* _pSystemStructure)
{
	// temporary system structure
	CSystemStructure* pNewSystemStructure = new CSystemStructure();
	pNewSystemStructure->SaveToFile(m_sFileName + "_temp.mdem");
	pNewSystemStructure->CreateFromSystemStructure(_pSystemStructure, 0);
	pNewSystemStructure->ClearAllTDData();

	// save models and simulation info
	ProtoModulesData* _pProtoMessage = _pSystemStructure->GetProtoModulesData();
	ProtoModulesData* _pProtoMessageNew = pNewSystemStructure->GetProtoModulesData();
	// save model manager data
	const ProtoModuleModelManager& MMOld = _pProtoMessage->model_manager();
	ProtoModuleModelManager* MMNEw = _pProtoMessageNew->mutable_model_manager();
	MMNEw->CopyFrom(MMOld);
	// save simulator data
	const ProtoModuleSimulator& simOld = _pProtoMessage->simulator();
	ProtoModuleSimulator* simNEw = _pProtoMessageNew->mutable_simulator();
	simNEw->CopyFrom(simOld);

	// get total number of objects in old file
	const size_t nNumberOfObjects = pNewSystemStructure->GetTotalObjectsCount();

	// get all time points from old file
	std::vector<double> vTimePoints = _pSystemStructure->GetAllTimePoints();
	if (vTimePoints.empty())
		vTimePoints = _pSystemStructure->GetAllTimePointsOldFormat();

	double dCurrTime = 0;
	bool IsFirstIter = true;
	for (auto i = 0; i < vTimePoints.size(); i++)
	{
		dCurrTime = vTimePoints[i];
		// in cases when first saved time point != 0 -> add zero time point, this action is to be checked and removed if it is unnecessary
		if (i == 0 && vTimePoints[0] != 0 && IsFirstIter)
		{
			i--;
			dCurrTime = 0;
			IsFirstIter = false;
		}
		m_sProgressMessage = "In progress... Current time point " + std::to_string(dCurrTime) + " [s]";

		for (auto j = 0; j < nNumberOfObjects; j++)
		{
			CPhysicalObject* pObject = _pSystemStructure->GetObjectByIndex(j);
			CPhysicalObject* pObjectNew = pNewSystemStructure->GetObjectByIndex(j);

			if (pObject && pObjectNew)
			{
				if (pObject->IsActive(dCurrTime))
				{
					switch (pObjectNew->GetObjectType())
					{
					case SPHERE:
					{
						pObjectNew->SetCoordinates(dCurrTime, pObject->GetCoordinates(dCurrTime));
						pObjectNew->SetVelocity(dCurrTime, pObject->GetVelocity(dCurrTime));
						pObjectNew->SetAngleVelocity(dCurrTime, pObject->GetAngleVelocity(dCurrTime));
						pObjectNew->SetForce(dCurrTime, pObject->GetForce(dCurrTime));
						pObjectNew->SetOrientation(dCurrTime, CQuaternion(pObject->GetAngles(dCurrTime)));
						break;
					}
					case SOLID_BOND:
					{
						CSolidBond* pSBond = dynamic_cast<CSolidBond*>(_pSystemStructure->GetObjectByIndex(j));
						CSolidBond* pSBondNew = dynamic_cast<CSolidBond*>(pNewSystemStructure->GetObjectByIndex(j));
						pSBondNew->SetForce(dCurrTime, pSBond->GetForce(dCurrTime));
						pSBondNew->SetTangentialOverlap(dCurrTime, pSBond->GetOldTangentialOverlap(dCurrTime));
						pSBondNew->SetTotalTorque(dCurrTime, pSBond->GetTotalTorque(dCurrTime));
						break;
					}
					case LIQUID_BOND:
					{
						CLiquidBond* pLBond = dynamic_cast<CLiquidBond*>(_pSystemStructure->GetObjectByIndex(j));
						CLiquidBond* pLBondNew = dynamic_cast<CLiquidBond*>(pNewSystemStructure->GetObjectByIndex(j));
						pLBondNew->SetForce(dCurrTime, pLBond->GetForce(dCurrTime));
						break;
					}
					case TRIANGULAR_WALL:
					{
						CTriangularWall* pWall = dynamic_cast<CTriangularWall*>(_pSystemStructure->GetObjectByIndex(j));
						CTriangularWall* pWallNew = dynamic_cast<CTriangularWall*>(pNewSystemStructure->GetObjectByIndex(j));
						pWallNew->SetPlaneCoord(dCurrTime, pWall->GetCoordVertex1(dCurrTime), pWall->GetOldCoordVertex2(dCurrTime), pWall->GetOldCoordVertex3(dCurrTime));
						pWallNew->SetForce(dCurrTime, pWall->GetForce(dCurrTime));
						pWallNew->SetVelocity(dCurrTime, pWall->GetVelocity(dCurrTime));
						break;
					}
					default: break;
					}
				}
				else
				{
					double dActivityStart, dActivityEnd;
					pObjectNew->GetActivityTimeInterval(&dActivityStart, &dActivityEnd);
					if (dActivityEnd > dCurrTime)
						pObjectNew->SetEndActivityTime(dCurrTime);
				}
			}
			m_dProgressPercent = double(i *  nNumberOfObjects + j) * 100 / (nNumberOfObjects * vTimePoints.size());
		}
	}

	// remove old ss and file
	delete _pSystemStructure;
	MUSENFileFunctions::removeFile(m_sFileName);

	// save new file with old file name
	pNewSystemStructure->SaveToFile(m_sFileName);

	// remove temporary ss and file
	delete pNewSystemStructure;
	MUSENFileFunctions::removeFile(m_sFileName + "_temp.mdem");

	m_sProgressMessage = "Transformation finished";
	m_dProgressPercent = 100;
}

void CFileConverter::ConvertFileV1ToV2(CSystemStructure* _pSystemStructure)
{
	m_dProgressPercent = 50;

	// temporary system structure
	_pSystemStructure->SaveToFile(m_sFileName + "_temp.mdem");

	// remove old file
	MUSENFileFunctions::removeFile(m_sFileName);

	// remove temporary ss
	delete _pSystemStructure;

	// rename saved file
	MUSENFileFunctions::renameFile(m_sFileName + "_temp.mdem", m_sFileName);

	m_sProgressMessage = "Transformation finished";
	m_dProgressPercent = 100;
}
