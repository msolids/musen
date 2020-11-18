/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ImportFromText.h"

CImportFromText::CImportFromText(CSystemStructure* _pSystemStructure)
{
	m_pSystemStructure = _pSystemStructure;
}

CImportFromText::EImportFileResult CImportFromText::CheckConstantProperties(ETXTCommands _identifier, const SRequiredProps& _props)
{
	if (_identifier == ETXTCommands::OBJECT_ID)	      return EImportFileResult::OK;
	if(!_props.bObjectID)		                      return EImportFileResult::ErrorNoID;

	if (_identifier == ETXTCommands::OBJECT_TYPE)     return EImportFileResult::OK;
	if (!_props.bObjectType)		                  return EImportFileResult::ErrorNoType;

	if (_identifier == ETXTCommands::OBJECT_GEOMETRY) return EImportFileResult::OK;
	if (!_props.bObjectGeometry)		              return EImportFileResult::ErrorNoGeometry;

	return EImportFileResult::OK;
}

void CImportFromText::SetAllTDdataIntoSystemStructure()
{
	for(double dTime : m_allTimePoints)
		for (size_t j = 0; j < m_pSystemStructure->GetTotalObjectsCount(); ++j)
		{
			CPhysicalObject* pCurrObject = m_pSystemStructure->GetObjectByIndex(j);
			if (!pCurrObject || !m_vObjects[j]) continue;
			switch (pCurrObject->GetObjectType())
			{
			case SPHERE:
			{
				pCurrObject->SetCoordinates(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vCoordinates, dTime));
				pCurrObject->SetVelocity(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vVelocity, dTime));
				pCurrObject->SetAngleVelocity(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vAngleVelocity, dTime));
				pCurrObject->SetForce(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vForce, dTime));
				pCurrObject->SetOrientation(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vQuaternion, dTime));
				pCurrObject->SetStressTensor(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vStressTensor, dTime));
				pCurrObject->SetTemperature(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vTemperature, dTime));
				break;
			}
			case SOLID_BOND:
			{
				pCurrObject->SetForce(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vForce, dTime));
				pCurrObject->SetAngleVelocity(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vAngleVelocity, dTime));	// tangential overlap
				std::vector<double> vTotalTorques;
				for (auto& coord : m_vObjects[j]->vCoordinates)
					vTotalTorques.push_back(coord.x);
				pCurrObject->SetTotalTorque(dTime, InterpolatedValue(m_vObjects[j]->vTime, vTotalTorques, dTime));
				pCurrObject->SetTemperature(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vTemperature, dTime));
				break;
			}
			case LIQUID_BOND:
			{
				pCurrObject->SetForce(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vForce, dTime));
				break;
			}
			case TRIANGULAR_WALL:
			{	pCurrObject->SetForce(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vForce, dTime));
				pCurrObject->SetVelocity(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vVelocity, dTime));
				pCurrObject->SetCoordinates(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vCoordinates, dTime));		// coordinate X
				pCurrObject->SetOrientation(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vQuaternion, dTime));			// coordinate Y
				pCurrObject->SetAngleVelocity(dTime, InterpolatedValue(m_vObjects[j]->vTime, m_vObjects[j]->vAngleVelocity, dTime));	// coordinate Z
				break;
			}
			default: break;
			}
		}
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
}

CImportFromText::SImportFileInfo CImportFromText::Import(const std::string& _fileName)
{
	SImportFileInfo status{ EImportFileResult::ErrorOpening, 0, false, false, false };

	// prepare initial mdem file
	m_pSystemStructure->ClearAllStatesFrom(0);
	m_pSystemStructure->DeleteAllObjects();
	m_pSystemStructure->ClearAllData();

	// open txt file
	std::ifstream inputFile;
	inputFile.open(UnicodePath(_fileName));
	if (!inputFile)
		return status;

	std::string sCurrentLine;	// current line
	while (safeGetLine(inputFile, sCurrentLine).good())
	{
		status.nErrorLineNumber++;
		if (sCurrentLine.empty()) continue;
		std::stringstream tempStream;
		tempStream << std::scientific << sCurrentLine;

		// corresponding flag is set to true if the property is specified in file; if not - import should be stopped and critical error message should be shown
		SRequiredProps requiredProps{ false, false, false };

		size_t nCurrentObjectID = 0;
		while (tempStream.good())
		{
			auto nIdentifier = static_cast<ETXTCommands>(GetValueFromStream<unsigned>(&tempStream));
			switch (nIdentifier)
			{
			// constant properties
			case ETXTCommands::OBJECT_ID:
				nCurrentObjectID = static_cast<size_t>(GetValueFromStream<double>(&tempStream)); // to handle scientific notation properly
				requiredProps.bObjectID = true;
				break;
			case ETXTCommands::OBJECT_TYPE:
			{
				status.importResult = CheckConstantProperties(nIdentifier, requiredProps);
				if (status.importResult != EImportFileResult::OK)
					return status;
				m_pSystemStructure->AddObject(GetValueFromStream<unsigned>(&tempStream), nCurrentObjectID);
				while (nCurrentObjectID >= m_vObjects.size())
					m_vObjects.push_back(nullptr);
				m_vObjects[nCurrentObjectID] = new STDObjectInfo;
				requiredProps.bObjectType = true;
				break;
			}
			case ETXTCommands::OBJECT_GEOMETRY:
			{
				status.importResult = CheckConstantProperties(nIdentifier, requiredProps);
				if (status.importResult != EImportFileResult::OK)
					return status;
				if (CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(nCurrentObjectID))
				{
					pObject->SetObjectGeometryText(tempStream);
					requiredProps.bObjectGeometry = true;
				}
				break;
			}
			case ETXTCommands::OBJECT_COMPOUND_TYPE:
			{
				status.importResult = CheckConstantProperties(nIdentifier, requiredProps);
				if (status.importResult != EImportFileResult::OK)
					return status;
				const std::string compoundKey = GetValueFromStream<std::string>(&tempStream);
				if (CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(nCurrentObjectID))
				{
					pObject->SetCompoundKey(compoundKey);
					status.bMaterial = true;
				}
				break;
			}
			case ETXTCommands::OBJECT_ACTIVITY_INTERVAL:
			{
				status.importResult = CheckConstantProperties(nIdentifier, requiredProps);
				if (status.importResult != EImportFileResult::OK)
					return status;
				auto dStartActivity = GetValueFromStream<double>(&tempStream);
				auto dEndActivity   = GetValueFromStream<double>(&tempStream);
				if (CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(nCurrentObjectID))
				{
					pObject->SetStartActivityTime(dStartActivity);
					pObject->SetEndActivityTime(dEndActivity);
					status.bActivityInterval = true;
				}
				break;
			}
			// TD properties
			case ETXTCommands::OBJECT_TIME:
			{
				status.importResult = CheckConstantProperties(nIdentifier, requiredProps);
				if (status.importResult != EImportFileResult::OK)
					return status;
				const auto dCurrentTime = GetValueFromStream<double>(&tempStream);
				m_allTimePoints.insert(dCurrentTime);
				m_vObjects[nCurrentObjectID]->vTime.push_back(dCurrentTime);
				m_vObjects[nCurrentObjectID]->vCoordinates.emplace_back(0, 0, 0);
				m_vObjects[nCurrentObjectID]->vVelocity.emplace_back(0, 0, 0);
				m_vObjects[nCurrentObjectID]->vAngleVelocity.emplace_back(0, 0, 0);
				m_vObjects[nCurrentObjectID]->vTotalForce.push_back(0);
				m_vObjects[nCurrentObjectID]->vForce.emplace_back(0, 0, 0);
				m_vObjects[nCurrentObjectID]->vQuaternion.emplace_back(0, 1, 0, 0);
				m_vObjects[nCurrentObjectID]->vStressTensor.emplace_back(0);
				m_vObjects[nCurrentObjectID]->vTemperature.emplace_back(0);
				break;
			}
			case ETXTCommands::OBJECT_COORDINATES:
				if (m_pSystemStructure->GetObjectByIndex(nCurrentObjectID)->GetObjectType() == SPHERE)
					status.bParticleCoordinates = true;
				tempStream >> m_vObjects[nCurrentObjectID]->vCoordinates.back(); break;
			case ETXTCommands::OBJECT_VELOCITIES:   tempStream >> m_vObjects[nCurrentObjectID]->vVelocity.back();      break;
			case ETXTCommands::OBJECT_ANG_VEL:      tempStream >> m_vObjects[nCurrentObjectID]->vAngleVelocity.back(); break;
			case ETXTCommands::OBJECT_TOTAL_FORCE:  tempStream >> m_vObjects[nCurrentObjectID]->vTotalForce.back();    break;
			case ETXTCommands::OBJECT_FORCE:        tempStream >> m_vObjects[nCurrentObjectID]->vForce.back();         break;
			case ETXTCommands::OBJECT_QUATERNION:   tempStream >> m_vObjects[nCurrentObjectID]->vQuaternion.back();    break;
			case ETXTCommands::OBJECT_STRESSTENSOR: tempStream >> m_vObjects[nCurrentObjectID]->vStressTensor.back();  break;
			case ETXTCommands::OBJECT_TEMPERATURE:  tempStream >> m_vObjects[nCurrentObjectID]->vTemperature.back();   break;
			case ETXTCommands::OBJECT_ANGL:       GetValueFromStream<CVector3>(&tempStream); break; // skip it
			case ETXTCommands::OBJECT_ACCEL:      GetValueFromStream<CVector3>(&tempStream); break; // skip it
			case ETXTCommands::OBJECT_ANGL_ACCEL: GetValueFromStream<CVector3>(&tempStream); break; // skip it
			// import information about scene
			case ETXTCommands::SIMULATION_DOMAIN:
				m_pSystemStructure->SetSimulationDomain(GetValueFromStream<SVolumeType>(&tempStream));
				break;
			case ETXTCommands::PERIODIC_BOUNDARIES:
			{
				CVector3 domainBeg, domainEnd;
				SPBC tempPBC;
				tempStream >> tempPBC.bEnabled >> tempPBC.bX >> tempPBC.bY >> tempPBC.bZ >> domainBeg >> domainEnd;
				tempPBC.SetDomain(domainBeg, domainEnd);
				tempPBC.vVel.Init(0);
				m_pSystemStructure->SetPBC(tempPBC);
				break;
			}
			case ETXTCommands::ANISOTROPY:
				m_pSystemStructure->EnableAnisotropy(GetValueFromStream<bool>(&tempStream));
				break;
			case ETXTCommands::CONTACT_RADIUS:
				m_pSystemStructure->EnableContactRadius(GetValueFromStream<bool>(&tempStream));
				break;
			// info about geometries
			case ETXTCommands::GEOMETRY:
			{
				CRealGeometry* pGeometry = m_pSystemStructure->AddGeometry();
				pGeometry->SetName(GetValueFromStream<std::string>(&tempStream));
				pGeometry->SetKey(GetValueFromStream<std::string>(&tempStream));
				pGeometry->SetMass(GetValueFromStream<double>(&tempStream));
				pGeometry->SetFreeMotion(GetValueFromStream<CBasicVector3<bool>>(&tempStream));
				break;
			}
			case ETXTCommands::GEOMETRY_PLANES:
			{
				CRealGeometry* pGeometry = m_pSystemStructure->Geometry(m_pSystemStructure->GeometriesNumber() - 1);
				std::vector<size_t> planes(GetValueFromStream<size_t>(&tempStream));
				for (auto& plane : planes)
					tempStream >> plane;
				pGeometry->SetPlanesIndices(planes);
				break;
			}
			case ETXTCommands::GEOMETRY_TDVEL:
			{
				CRealGeometry* pGeometry = m_pSystemStructure->Geometry(m_pSystemStructure->GeometriesNumber() - 1);
				const size_t nIntervals = GetValueFromStream<unsigned>(&tempStream);
				pGeometry->Motion()->Clear();
				double prevTime = 0, time;
				CVector3 vel, rotCenter, rotVel;
				for (size_t i = 0; i < nIntervals; ++i)
				{
					tempStream >> time >> vel >> rotCenter >> rotVel;
					pGeometry->Motion()->AddTimeInterval(CGeometryMotion::STimeMotionInterval{ prevTime, time, CGeometryMotion::SMotionInfo{vel, rotVel, rotCenter} });
					prevTime = time;
				}
				break;
			}
			// info about materials
			case ETXTCommands::MATERIALS_COMPOUNDS:
			{
				CCompound* pCompound = new CCompound();
				pCompound->SetKey(GetValueFromStream<std::string>(&tempStream));
				pCompound->SetName(GetValueFromStream<std::string>(&tempStream));
				while (tempStream.good())
				{
					auto prop = GetValueFromStream<unsigned>(&tempStream);
					auto value = GetValueFromStream<double>(&tempStream);
					pCompound->SetPropertyValue(prop, value);
				}
				m_pSystemStructure->m_MaterialDatabase.AddCompound(*pCompound);
				delete pCompound;
				break;
			}
			case ETXTCommands::MATERIALS_INTERACTIONS:
			{
				CInteraction* pInteraction = m_pSystemStructure->m_MaterialDatabase.GetInteraction(GetValueFromStream<std::string>(&tempStream), GetValueFromStream<std::string>(&tempStream));
				if (pInteraction)
					while (tempStream.good())
					{
						auto prop = GetValueFromStream<unsigned>(&tempStream);
						auto value = GetValueFromStream<double>(&tempStream);
						pInteraction->SetPropertyValue(prop, value);
					}
				break;
			}
			case ETXTCommands::MATERIALS_MIXTURES:
			{
				CMixture* pMixture = new CMixture();
				pMixture->SetKey(GetValueFromStream<std::string>(&tempStream));
				pMixture->SetName(GetValueFromStream<std::string>(&tempStream));
				while (tempStream.good())
				{
					size_t iFraction = pMixture->AddFraction();
					const auto nNumberOfFraction = GetValueFromStream<unsigned>(&tempStream);
					pMixture->SetFractionCompound(iFraction, GetValueFromStream<std::string>(&tempStream));
					pMixture->SetFractionDiameter(iFraction, GetValueFromStream<double>(&tempStream));
					pMixture->SetFractionValue(iFraction, GetValueFromStream<double>(&tempStream));
					pMixture->SetFractionName(iFraction, "Fraction " + std::to_string(nNumberOfFraction));
				}
				m_pSystemStructure->m_MaterialDatabase.AddMixture(*pMixture);
				delete pMixture;
				break;
			}
			}
		}
	}

	// sets all time-dependent data from local storage m_vObjects into system structure
	SetAllTDdataIntoSystemStructure();

	// free local storage
	for (auto& object : m_vObjects)
		delete object;

	status.importResult = EImportFileResult::OK;
	return status;
}