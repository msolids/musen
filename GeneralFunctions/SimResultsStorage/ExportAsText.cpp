/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ExportAsText.h"
#include <iomanip>

CExportAsText::CExportAsText() :
	m_pSystemStructure{ nullptr },
	m_pConstraints{ nullptr },
	m_fileName{ "" },
	m_precision{ std::cout.precision() },
	m_dProgressPercent{ 0 },
	m_sProgressMessage{ "" },
	m_sErrorMessage{ "" },
	m_nCurrentStatus{ ERunningStatus::IDLE }
{
	m_allFlags = { &m_objectTypeFlags, &m_sceneInfoFlags, &m_constPropsFlags, &m_tdPropsFlags, &m_geometriesFlags, &m_materialsFlags };
}

void CExportAsText::SetPointers(CSystemStructure* _pSystemStructure, CConstraints* _pConstaints)
{
	m_pSystemStructure = _pSystemStructure;
	m_pConstraints = _pConstaints;
}

void CExportAsText::SetFlags(const SObjectTypeFlags& _objectTypes, const SSceneInfoFlags& _sceneInfo, const SConstPropsFlags& _constProps, const STDPropsFlags& _tdProps, const SGeometriesFlags& _geometries, const SMaterialsFlags& _materials)
{
	m_objectTypeFlags = _objectTypes;
	m_sceneInfoFlags = _sceneInfo;
	m_constPropsFlags = _constProps;
	m_tdPropsFlags = _tdProps;
	m_geometriesFlags = _geometries;
	m_materialsFlags = _materials;
}

void CExportAsText::SetFileName(const std::string& _sFileName)
{
	m_fileName = _sFileName;
}

void CExportAsText::SetTimePoints(const std::vector<double>& _vTimePoints)
{
	m_timePoints = _vTimePoints;
}

void CExportAsText::SetPrecision(int _nPrecision)
{
	m_precision = _nPrecision;
}

double CExportAsText::GetProgressPercent() const
{
	return m_dProgressPercent;
}

const std::string& CExportAsText::GetProgressMessage() const
{
	return m_sProgressMessage;
}

const std::string& CExportAsText::GetErrorMessage() const
{
	return m_sErrorMessage;
}

void CExportAsText::SetCurrentStatus(const ERunningStatus& _nNewStatus)
{
	m_nCurrentStatus = _nNewStatus;
}

ERunningStatus CExportAsText::GetCurrentStatus() const
{
	return m_nCurrentStatus;
}

int CExportAsText::GetPrecision() const
{
	return static_cast<int>(m_precision);
}

std::set<size_t> CExportAsText::GetObjectsIDs() const
{
	std::set<size_t> setObjectsIDs;
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
		if (const CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(i))
			setObjectsIDs.insert(object->m_lObjectID);
	return setObjectsIDs;
}

std::vector<CExportAsText::SConstantData> CExportAsText::GetConstantProperties(const std::set<size_t>& _setObjectsIDs) const
{
	if (_setObjectsIDs.empty())	return {};

	std::vector<SConstantData> vConstData;
	vConstData.reserve(m_pSystemStructure->GetTotalObjectsCount());
	for (size_t id : _setObjectsIDs)
	{
		const CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(id);
		if (!object) continue;
		SConstantData tmpGenData;
		tmpGenData.objID = object->m_lObjectID;
		tmpGenData.objType = object->GetObjectType();
		tmpGenData.objGeom = object->GetObjectGeometryText();
		tmpGenData.cmpKey = object->GetCompoundKey().empty() ? "Undefined" : object->GetCompoundKey();
		object->GetActivityTimeInterval(&tmpGenData.activeStart, &tmpGenData.activeEnd);
		vConstData.push_back(tmpGenData);
	}
	return vConstData;
}

std::set<size_t> CExportAsText::CheckConstraints(const std::set<size_t>& _setObjectsIDs) const
{
	if (_setObjectsIDs.empty())	return {};

	// selected IDs for each type of object
	std::set<size_t> setPIDs, setSBIDs, setLBIDs, setTWIDs;
	setPIDs = setSBIDs = setLBIDs = setTWIDs = _setObjectsIDs;

	// check diameter constraints
	if (!m_pConstraints->IsAllDiametersSelected())
	{
		setPIDs  = m_pConstraints->ApplyDiameterFilter(m_timePoints[0], SPHERE,          &setPIDs);
		setSBIDs = m_pConstraints->ApplyDiameterFilter(m_timePoints[0], SOLID_BOND,      &setSBIDs);
		setLBIDs = m_pConstraints->ApplyDiameterFilter(m_timePoints[0], LIQUID_BOND,     &setLBIDs);
		setTWIDs = m_pConstraints->ApplyDiameterFilter(m_timePoints[0], TRIANGULAR_WALL, &setTWIDs);
	}
	// check material constraints
	if (!m_pConstraints->IsAllMaterialsSelected())
	{
		setPIDs  = m_pConstraints->ApplyMaterialFilter(m_timePoints[0], SPHERE,          &setPIDs);
		setSBIDs = m_pConstraints->ApplyMaterialFilter(m_timePoints[0], SOLID_BOND,      &setSBIDs);
		setLBIDs = m_pConstraints->ApplyMaterialFilter(m_timePoints[0], LIQUID_BOND,     &setLBIDs);
		setTWIDs = m_pConstraints->ApplyMaterialFilter(m_timePoints[0], TRIANGULAR_WALL, &setTWIDs);
	}
	// check analysis volume constraints
	if (!m_pConstraints->IsAllVolumeSelected())
		for (double t : m_timePoints)
		{
			if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;
			setPIDs  = m_pConstraints->ApplyVolumeFilter(t, SPHERE,          &setPIDs);
			setSBIDs = m_pConstraints->ApplyVolumeFilter(t, SOLID_BOND,      &setSBIDs);
			setLBIDs = m_pConstraints->ApplyVolumeFilter(t, LIQUID_BOND,     &setLBIDs);
			setTWIDs = m_pConstraints->ApplyVolumeFilter(t, TRIANGULAR_WALL, &setTWIDs);
		}

	// merge IDs of all selected objects
	std::set<size_t> setSelectedObjectsIDs;
	setSelectedObjectsIDs.insert(setPIDs.begin(),  setPIDs.end());
	setSelectedObjectsIDs.insert(setSBIDs.begin(), setSBIDs.end());
	setSelectedObjectsIDs.insert(setLBIDs.begin(), setLBIDs.end());
	setSelectedObjectsIDs.insert(setTWIDs.begin(), setTWIDs.end());
	return setSelectedObjectsIDs;
}

bool CExportAsText::SaveTDDataToBinFile(const std::string& _sFileName, const std::set<size_t>& _setObjectsIDs)
{
	if (_setObjectsIDs.empty()) return true;

	// create temporary file for time-dependent data
	std::ofstream binOut(UnicodePath(_sFileName), std::ios::binary);
	if (binOut.fail()) return false;

	// get time-dependent data and save into temporary binary file
	for (size_t i = 0; i < m_timePoints.size(); ++i)
	{
		int j = 0;
		for (size_t id : _setObjectsIDs)
		{
			if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;
			CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(id);
			if (!object) continue;
			STDData TDD;
			TDD.time = m_timePoints[i];
			TDD.coord = object->GetCoordinates(m_timePoints[i]);
			TDD.velo = object->GetVelocity(m_timePoints[i]);
			TDD.angleVel = object->GetAngleVelocity(m_timePoints[i]);
			TDD.totForce = object->GetForce(m_timePoints[i]).Length();
			TDD.force = object->GetForce(m_timePoints[i]);
			TDD.quaternion = object->GetOrientation(m_timePoints[i]);
			TDD.stressTensor = object->GetStressTensor(m_timePoints[i]);
			TDD.temperature = object->GetTemperature(m_timePoints[i]);
			if (object->GetObjectType() == SOLID_BOND)
				TDD.coord = CVector3(object->GetTotalTorque(m_timePoints[i]), 0, 0);

			const std::streampos pos = sizeof(STDData) * (m_timePoints.size() * j + i);
			binOut.seekp(pos);
			binOut.write(reinterpret_cast<char*>(&TDD), sizeof(TDD));

			m_dProgressPercent = double(i *  _setObjectsIDs.size() + j) * 100 / (m_timePoints.size() * _setObjectsIDs.size()) / 2;
			j++;
		}
	}
	// close temporary binary file
	binOut.close();
	return true;
}

void CExportAsText::Export()
{
	// set initial values
	m_dProgressPercent = 0;
	m_sErrorMessage = "";
	m_sProgressMessage = "Export started. Please wait...";
	m_nCurrentStatus = ERunningStatus::RUNNING;

	// check time points
	if (m_timePoints.empty())
	{
		m_sErrorMessage = "Error! There are no time points selected for saving.";
		return;
	}

	// get identifiers of all real objects
	std::set<size_t> setStartIDs = GetObjectsIDs();

	///////////////////////////////////////////////////////  APPLICATION OF CONSTRAINTS ////////////////////////////////////////////////////////////////////

	// check constraints
	if (!IsSaveAll())
	{
		if (m_objectTypeFlags.IsAllOff()) // no object types are selected
			setStartIDs.clear();
		else                              // some object types are selected
		{
			// remove unnecessary identifiers according to selected object type checkboxes
			if (!setStartIDs.empty() && !m_objectTypeFlags.IsAllOn()) // not all object types are selected
			{
				m_sProgressMessage = "Application of selected object types...";
				for (auto itCurrID = setStartIDs.begin(), itEndID = setStartIDs.end(); itCurrID != itEndID;)
				{
					const unsigned type = m_pSystemStructure->GetObjectByIndex(*itCurrID)->GetObjectType();
					if (type == SPHERE          && !m_objectTypeFlags.particles
					 || type == SOLID_BOND      && !m_objectTypeFlags.solidBonds
					 || type == LIQUID_BOND     && !m_objectTypeFlags.liquidBonds
					 || type == TRIANGULAR_WALL && !m_objectTypeFlags.triangularWalls)
						setStartIDs.erase(itCurrID++);
					else
						++itCurrID;
				}
			}
			// application of constraints
			if (!setStartIDs.empty() && (!m_pConstraints->IsAllDiametersSelected() || !m_pConstraints->IsAllMaterialsSelected() || !m_pConstraints->IsAllVolumeSelected()))
			{
				m_sProgressMessage = "Application of constraints...";
				setStartIDs = CheckConstraints(setStartIDs);
			}
		}
	}

	///////////////////////////////////////////////////////  DATA PREPARING  /////////////////////////////////////////////////////////////////////////////

	std::vector<SConstantData> vConstData;
	std::string sTmpBinFileName = m_fileName + ".exportAsTextTemp";
	if (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED)
	{
		// get constant properties for all selected objects
		m_sProgressMessage = "Preparation of constant properties...";
		vConstData = GetConstantProperties(setStartIDs);

		// get and save time-dependent data into temporary file
		if (!vConstData.empty())
		{
			m_sProgressMessage = "Preparation of time-dependent properties...";
			if (!SaveTDDataToBinFile(sTmpBinFileName, setStartIDs))
			{
				m_sErrorMessage = "Error! Cannot create a temporary binary file.";
				return;
			}
		}
	}

	///////////////////////////////////////////////////////  SAVE DATA INTO FINAL TEXT FILE  /////////////////////////////////////////////////////////////

	std::ofstream txtOutFile;
	if (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED)
	{
		// create final text file
		txtOutFile.open(UnicodePath(m_fileName));
		if (txtOutFile.fail())
		{
			m_sErrorMessage = "Error! Cannot open a final text file for writing.";
			return;
		}

		// save constant and time-dependent data into final text file
		m_sProgressMessage = "Export data to text file...";
		if (!vConstData.empty())
		{
			// open temporary binary file for reading
			std::ifstream binInFile(UnicodePath(sTmpBinFileName), std::ios::binary);
			if (binInFile.fail())
			{
				m_sErrorMessage = "Error! Cannot open a temporary binary file for reading.";
				return;
			}

			for (size_t i = 0; i < vConstData.size(); ++i)
			{
				if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;

				// write constant information
				if (m_constPropsFlags.id)
					txtOutFile        << E2I(ETXTCommands::OBJECT_ID)                << " " << vConstData[i].objID;
				if (m_constPropsFlags.type)
					txtOutFile << " " << E2I(ETXTCommands::OBJECT_TYPE)              << " " << vConstData[i].objType;
				if (m_constPropsFlags.geometry)
					txtOutFile << " " << E2I(ETXTCommands::OBJECT_GEOMETRY)          << " " << vConstData[i].objGeom;
				if (m_constPropsFlags.material)
					txtOutFile << " " << E2I(ETXTCommands::OBJECT_COMPOUND_TYPE)     << " " << vConstData[i].cmpKey;
				if (m_constPropsFlags.activityInterval)
					txtOutFile << " " << E2I(ETXTCommands::OBJECT_ACTIVITY_INTERVAL) << " " << vConstData[i].activeStart << " " << vConstData[i].activeEnd;

				// set new precision and save default for TD data
				std::streamsize defaultPrecision = txtOutFile.precision();
				txtOutFile.precision(m_precision);

				// write time-dependent data into final text file
				for (size_t j = 0; j < m_timePoints.size(); ++j)
				{
					if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED) break;

					STDData TDD;
					const std::streampos pos = sizeof(STDData) * (m_timePoints.size() * i + j);
					binInFile.seekg(pos);
					binInFile.read(reinterpret_cast<char*>(&TDD), sizeof(TDD));

					txtOutFile << " " << E2I(ETXTCommands::OBJECT_TIME) << " " << m_timePoints[j];
					if (m_tdPropsFlags.coordinate        && vConstData[i].objType == SPHERE
					 || m_tdPropsFlags.planesCoordinates && vConstData[i].objType == TRIANGULAR_WALL
					 || m_tdPropsFlags.totalTorque       && vConstData[i].objType == SOLID_BOND)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_COORDINATES)  << " " << TDD.coord;
					if (m_tdPropsFlags.velocity)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_VELOCITIES)   << " " << TDD.velo;
					if (m_tdPropsFlags.angularVelocity   &&  vConstData[i].objType == SPHERE
					 || m_tdPropsFlags.planesCoordinates &&  vConstData[i].objType == TRIANGULAR_WALL
					 || m_tdPropsFlags.tangOverlap       &&  vConstData[i].objType == SOLID_BOND)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_ANG_VEL)      << " " << TDD.angleVel;
					if (m_tdPropsFlags.totalForce)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_TOTAL_FORCE)  << " " << TDD.totForce;
					if (m_tdPropsFlags.force)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_FORCE)        << " " << TDD.force;
					if (m_tdPropsFlags.quaternion        && vConstData[i].objType == SPHERE
					 || m_tdPropsFlags.planesCoordinates && vConstData[i].objType == TRIANGULAR_WALL)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_QUATERNION)   << " " << TDD.quaternion;
					if (m_tdPropsFlags.stressTensor      && vConstData[i].objType == SPHERE)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_STRESSTENSOR) << " " << TDD.stressTensor;
					if (m_tdPropsFlags.temperature       && vConstData[i].objType == SPHERE
					 || m_tdPropsFlags.temperature       && vConstData[i].objType == SOLID_BOND)
						txtOutFile << " " << E2I(ETXTCommands::OBJECT_TEMPERATURE)  << " " << TDD.temperature;

					m_dProgressPercent = 50 + double(i *  m_timePoints.size() + j) * 100 / (m_timePoints.size() * setStartIDs.size()) / 2;
				}
				txtOutFile << std::endl;

				// set back default precision
				txtOutFile.precision(defaultPrecision);
			}
			// close temporary binary file
			binInFile.close();
		}
	}
	// remove temporary binary file
	std::remove(sTmpBinFileName.c_str());

	// save information about scene
	if (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED)
	{
		// simulation domain
		if (m_sceneInfoFlags.domain)
			txtOutFile << E2I(ETXTCommands::SIMULATION_DOMAIN) << " " << m_pSystemStructure->GetSimulationDomain().coordBeg << " "
			<< m_pSystemStructure->GetSimulationDomain().coordEnd << std::endl;

		// periodic boundary conditions
		if (m_sceneInfoFlags.pbc)
			txtOutFile << E2I(ETXTCommands::PERIODIC_BOUNDARIES) << " " << m_pSystemStructure->GetPBC().bEnabled << " "
			<< m_pSystemStructure->GetPBC().bX << " " << m_pSystemStructure->GetPBC().bY << " " << m_pSystemStructure->GetPBC().bZ << " "
			<< m_pSystemStructure->GetPBC().initDomain.coordBeg << " " << m_pSystemStructure->GetPBC().initDomain.coordEnd << std::endl;

		// consideration of anisotropy
		if (m_sceneInfoFlags.anisotropy)
			txtOutFile << E2I(ETXTCommands::ANISOTROPY) << " " << m_pSystemStructure->IsAnisotropyEnabled() << std::endl;

		// consideration of contact radius of particles
		if (m_sceneInfoFlags.contactRadius)
			txtOutFile << E2I(ETXTCommands::CONTACT_RADIUS) << " " << m_pSystemStructure->IsContactRadiusEnabled() << std::endl;
	}

	// save info about geometries
	if (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED)
	{
		for (size_t i = 0; i < m_pSystemStructure->GetGeometriesNumber(); ++i)
		{
			const SGeometryObject* geometry = m_pSystemStructure->GetGeometry(i);
			if (!geometry) continue;
			if (m_geometriesFlags.baseInfo)
				txtOutFile << E2I(ETXTCommands::GEOMETRY) << " " << geometry->sName << " " << geometry->sKey << " "
			    << geometry->dMass << " " << geometry->vFreeMotion << std::endl;
			if (m_geometriesFlags.tdProperties)
			{
				txtOutFile << E2I(ETXTCommands::GEOMETRY_TDVEL) << " " << geometry->vIntervals.size();
				for (const auto& interval : geometry->vIntervals)
					txtOutFile << " " << interval.dCriticalValue << " " << interval.vVel << " "	<< interval.vRotCenter << " " << interval.vRotVel;
				txtOutFile << std::endl;
			}
			if (m_geometriesFlags.wallsList)
			{
				txtOutFile << E2I(ETXTCommands::GEOMETRY_PLANES) << " " << geometry->vPlanes.size();
				for (size_t plane : geometry->vPlanes)
					txtOutFile << " " << plane;
				txtOutFile << std::endl;
			}
		}
	}

	// save info about materials
	if (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED)
	{
		// compounds
		if (m_materialsFlags.compounds)
		{
			std::vector<unsigned> vMusenActiveProperties = _MUSEN_ACTIVE_PROPERTIES;
			const size_t nCompundsNumber = m_pSystemStructure->m_MaterialDatabase.CompoundsNumber();
			for (size_t i = 0; i < nCompundsNumber; i++)
			{
				const CCompound* pCompound = m_pSystemStructure->m_MaterialDatabase.GetCompound(i);

				txtOutFile << E2I(ETXTCommands::MATERIALS_COMPOUNDS) << " " << pCompound->GetKey() << " " << pCompound->GetName();

				for (unsigned int property : vMusenActiveProperties)
					txtOutFile << " " << property << " " << pCompound->GetProperty(property)->GetValue();
				txtOutFile << std::endl;
			}
		}
		// interactions
		if (m_materialsFlags.interactions)
		{
			std::vector<unsigned> vMusenActiveInteractions = _MUSEN_ACTIVE_INTERACTIONS;
			const size_t nInteractionsNumber = m_pSystemStructure->m_MaterialDatabase.InteractionsNumber();
			for (size_t i = 0; i < nInteractionsNumber; i++)
			{
				const CInteraction* pInteraction = m_pSystemStructure->m_MaterialDatabase.GetInteraction(i);

				txtOutFile << E2I(ETXTCommands::MATERIALS_INTERACTIONS) << " " << pInteraction->GetKey1() << " " << pInteraction->GetKey2();

				for (unsigned int interaction : vMusenActiveInteractions)
					txtOutFile << " " << interaction << " " << pInteraction->GetProperty(interaction)->GetValue();
				txtOutFile << std::endl;
			}
		}
		// mixtures
		if (m_materialsFlags.mixtures)
		{
			size_t nMixturesNumber = m_pSystemStructure->m_MaterialDatabase.MixturesNumber();
			for (size_t i = 0; i < nMixturesNumber; i++)
			{
				CMixture* pMixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(i);

				txtOutFile << E2I(ETXTCommands::MATERIALS_MIXTURES) << " " << pMixture->GetKey() << " " << pMixture->GetName();

				size_t nFractionsNumber = pMixture->FractionsNumber();
				for (size_t j = 0; j < nFractionsNumber; j++)
					txtOutFile << " " << j << " " << pMixture->GetFractionCompound(j) << " " << pMixture->GetFractionDiameter(j) << " " << pMixture->GetFractionValue(j);
				txtOutFile << std::endl;
			}
		}
	}
	// close final text file
	txtOutFile << std::endl;
	txtOutFile.close();

	if (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED)
		m_dProgressPercent = 100;
	m_nCurrentStatus = ERunningStatus::IDLE;
	m_sProgressMessage = "Export finished.";
}

bool CExportAsText::IsSaveAll() const
{
	for (auto flags : m_allFlags)
		if (!flags->IsAllOn())
			return false;
	return true;
}
