/* Copyright (c) 2013-2022, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ExportAsText.h"
#include "Constraints.h"
#include "PackageGenerator.h"
#include "BondsGenerator.h"
#include "MUSENFileFunctions.h"

void CExportAsText::SetPointers(const CSystemStructure* _systemStructure, const CConstraints* _constraints, const CPackageGenerator* _pakageGenerator, const CBondsGenerator* _bondsGenerator)
{
	m_systemStructure  = _systemStructure;
	m_constraints      = _constraints;
	m_packageGenerator = _pakageGenerator;
	m_bondsGenerator   = _bondsGenerator;
}

void CExportAsText::SetSelectors(const SExportSelector& _selectors)
{
	m_selectors = _selectors;
}

void CExportAsText::SetFileName(const std::filesystem::path& _name)
{
	m_resFileName = _name;
	// construct the name of a temporary binary file
	m_tdData.binFileName = _name;
	m_tdData.binFileName.replace_extension("temp_musen_export");
}

void CExportAsText::SetTimePoints(const std::vector<double>& _timePoints)
{
	m_timePoints = _timePoints;
}

void CExportAsText::SetPrecision(int _precision)
{
	m_precision = _precision;
}

int CExportAsText::GetPrecision() const
{
	return static_cast<int>(m_precision);
}

ERunningStatus CExportAsText::GetStatus() const
{
	return m_status;
}

double CExportAsText::GetProgress() const
{
	return m_progress;
}

std::string CExportAsText::GetStatusMessage() const
{
	return m_statusMessage;
}

void CExportAsText::RequestStop()
{
	SetStatus("Stopped by user", ERunningStatus::TO_BE_STOPPED);
}

void CExportAsText::RequestErrorStop(const std::string& _message)
{
	SetStatus("Error: " + _message, ERunningStatus::FAILED);
}

void CExportAsText::SetStatus(const std::string& _message, ERunningStatus _status/* = ERunningStatus::RUNNING*/)
{
	m_statusMessage = _message;
	m_status = _status;
}

void CExportAsText::Export()
{
	// set initial values
	SetStatus("Export started. Please wait", ERunningStatus::RUNNING);
	m_progress = 0;
	m_constData.clear();
	m_tdData.tpActive.clear();
	m_tdData.binFileLayout.clear();

	// check file name
	if (m_resFileName.empty())
		RequestErrorStop("No output file selected!");
	if (ToBeStopped()) return Finalize();

	// try open output file
	TryOpenFileW(m_resFile, m_resFileName);
	if (ToBeStopped()) return Finalize();

	// check time points
	if (m_timePoints.empty() && !m_selectors.objectTypes.AllOff() && (!m_selectors.tdPropsPart.AllOff() || !m_selectors.tdPropsBond.AllOff() || !m_selectors.tdPropsWall.AllOff()))
		RequestErrorStop("No time points selected!");
	if (ToBeStopped()) return Finalize();

	// get ids of objects that have to be exported
	const std::set<size_t> IDs = FilterObjects(Vector2Set(m_systemStructure->GetAllObjectsIDs()));
	if (ToBeStopped()) return Finalize();

	// prepare constant data for filtered objects
	PrepareConstData(IDs);
	if (ToBeStopped()) return Finalize();

	// prepare time-dependent data for filtered objects
	PrepareTDData();
	if (ToBeStopped()) return Finalize();

	// write data
	WriteObjectsData();
	if (ToBeStopped()) return Finalize();
	WriteSceneData();
	if (ToBeStopped()) return Finalize();
	WriteGeometriesData();
	if (ToBeStopped()) return Finalize();
	WriteMaterialsData();
	if (ToBeStopped()) return Finalize();
	WriteGeneratorsData();
	if (ToBeStopped()) return Finalize();

	// finalize
	SetStatus("Export finished", ERunningStatus::IDLE);
	Finalize();
}

std::set<size_t> CExportAsText::FilterObjects(const std::set<size_t>& _ids)
{
	if (_ids.empty()) return {};

	// everything selected
	if (m_selectors.AllOn()) return _ids;

	std::set<size_t> filtered = _ids;
	filtered = FilterObjectsByType(filtered);
	filtered = FilterObjectsByActivity(filtered);
	filtered = FilterObjectsByMaterial(filtered);
	filtered = FilterObjectsByDiameter(filtered);
	filtered = FilterObjectsByVolume(filtered);
	return filtered;
}

std::set<size_t> CExportAsText::FilterObjectsByType(const std::set<size_t>& _ids)
{
	if (_ids.empty()) return {};

	// no types selected
	if (m_selectors.objectTypes.AllOff()) return {};
	// all types selected
	if (m_selectors.objectTypes.AllOn()) return _ids;

	SetStatus("Filtering objects by type");

	std::set<size_t> res;
	for (const size_t id : _ids)
	{
		const auto type = m_systemStructure->GetObjectByIndex(id)->GetObjectType();
		if (   type == SPHERE          && m_selectors.objectTypes.particles
			|| type == SOLID_BOND      && m_selectors.objectTypes.bonds
			|| type == TRIANGULAR_WALL && m_selectors.objectTypes.walls)
			res.insert(res.end(), id);
	}
	return res;
}

std::set<size_t> CExportAsText::FilterObjectsByActivity(const std::set<size_t>& _ids)
{
	// TODO: fill in m_tdFlags
	if (_ids.empty()) return {};

	// if no time-dependent properties selected, pay no attention to activity
	if (m_selectors.tdPropsPart.AllOff() && m_selectors.tdPropsBond.AllOff() && m_selectors.tdPropsWall.AllOff()) return _ids;

	SetStatus("Filtering objects by activity");

	// usually very few or none objects are filtered out here, so do it through deletion
	std::set<size_t> res = _ids;
	for (const size_t id : _ids)
	{
		const auto [intervalBeg, intervalEnd] = m_systemStructure->GetObjectByIndex(id)->GetActivityTimeInterval();
		if (m_timePoints.back() < intervalBeg || m_timePoints.front() > intervalEnd)
			res.erase(id);
	}

	return res;
}

std::set<size_t> CExportAsText::FilterObjectsByMaterial(const std::set<size_t>& _ids)
{
	if (m_constraints->IsAllMaterialsSelected()) return _ids;

	SetStatus("Filtering objects by material");

	return m_constraints->ApplyMaterialFilter(_ids);
}

std::set<size_t> CExportAsText::FilterObjectsByDiameter(const std::set<size_t>& _ids)
{
	if (m_constraints->IsAllDiametersSelected()) return _ids;

	SetStatus("Filtering objects by diameter");

	return m_constraints->ApplyDiameterFilter(_ids);
}

std::set<size_t> CExportAsText::FilterObjectsByVolume(const std::set<size_t>& _ids)
{
	if (m_constraints->IsAllVolumeSelected()) return _ids;

	SetStatus("Filtering objects by volume");

	// prepare time-dependent activity flags
	m_tdData.tpActive.clear();

	std::set<size_t> res;
	for (size_t iTime = 0; iTime < m_timePoints.size(); ++iTime)
	{
		if (ToBeStopped()) break;

		SetStatus("Filtering objects by volume (time point " + std::to_string(iTime + 1) + "/" + std::to_string(m_timePoints.size()) + ")");

		std::set<size_t> filtered = m_constraints->ApplyVolumeFilter(_ids, m_timePoints[iTime]);
		res = SetUnion(res, filtered);

		// fill time-dependent activity flags
		for (const auto id : filtered)
		{
			m_tdData.tpActive[id].resize(m_timePoints.size()); // if not exist yet, create the vector and resize it
			m_tdData.tpActive[id][iTime] = true;
		}

		m_progress = static_cast<double>(iTime) * 100 / static_cast<double>(m_timePoints.size());
	}

	return res;
}

void CExportAsText::PrepareConstData(const std::set<size_t>& _objectsID)
{
	if (_objectsID.empty())	return;

	SetStatus("Gathering constant object properties");

	m_constData.reserve(_objectsID.size());
	size_t iObj = 1;
	for (const size_t id : _objectsID)
	{
		SetStatus("Gathering constant object properties (object " + std::to_string(iObj) + "/" + std::to_string(_objectsID.size()) + ")");

		const CPhysicalObject* object = m_systemStructure->GetObjectByIndex(id);
		// TODO: write only required or selected values, discard the rest
		m_constData.emplace_back(
			  object->m_lObjectID
			, object->GetObjectType()
			, object->GetObjectGeometryText()
			, object->GetCompoundKey().empty() ? "Undefined" : object->GetCompoundKey()
			, object->GetActivityTimeInterval()
		);

		m_progress = static_cast<double>(iObj++) * 100 / static_cast<double>(_objectsID.size());
	}
}

void CExportAsText::PrepareTDData()
{
	if (m_constData.empty() || m_timePoints.empty()) return;

	SetStatus("Gathering time-dependent object properties");

	TryOpenFileW(m_tdData.binFileW, m_tdData.binFileName, std::ios::binary);
	if (ToBeStopped()) return;

	// prepare file layout information
	CalculateBinFileLayout();
	auto& lt = m_tdData.binFileLayout; // alias for layout

	// resize temporary file to fit all data
	try
	{
		std::filesystem::resize_file(m_tdData.binFileName, lt[SPHERE].fullLen + lt[SOLID_BOND].fullLen + lt[TRIANGULAR_WALL].fullLen);
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		return RequestErrorStop("Can not resize temporary file! " + std::string(e.what()));
	}

	// get time-dependent data and save into temporary binary file
	for (size_t iTime = 0; iTime < m_timePoints.size(); ++iTime)
	{
		if (ToBeStopped()) break;

		SetStatus("Gathering time-dependent object properties (time point " + std::to_string(iTime + 1) + "/" + std::to_string(m_timePoints.size()) + ")");

		std::map<unsigned, size_t> counter{{SPHERE, 0}, {SOLID_BOND, 0}, {TRIANGULAR_WALL, 0}}; // counters for already processed objects
		m_systemStructure->PrepareTimePointForRead(m_timePoints[iTime]);
		for (size_t iObj = 0; iObj < m_constData.size(); ++iObj)
		{
			if (ToBeStopped()) break;

			// get selected time-dependent data
			const CByteStream stream = GetBinData(iObj);

			// write data to temporary binary file
			const auto& type = m_constData[iObj].type;
			const std::streampos pos = lt[type].startPos + lt[type].entryLen * (m_timePoints.size() * counter[type]++ + iTime);
			m_tdData.binFileW.seekp(pos);
			m_tdData.binFileW.write(&stream.GetDataRef()[0], stream.Size());

			// update progress
			m_progress = static_cast<double>(iTime * m_constData.size() + iObj) * 100 / (m_timePoints.size() * m_constData.size());
		}
	}
	// close temporary binary file
	m_tdData.binFileW.close();
}

void CExportAsText::WriteObjectsData()
{
	if (m_constData.empty()) return;

	SetStatus("Writing objects data to result file");

	TryOpenFileR(m_tdData.binFileR, m_tdData.binFileName, std::ios::binary);
	if (ToBeStopped()) return;

	auto& lt = m_tdData.binFileLayout; // alias for layout
	std::map<unsigned, size_t> counter{ {SPHERE, 0}, {SOLID_BOND, 0}, {TRIANGULAR_WALL, 0} }; // counters for already processed objects
	for (size_t iObj = 0; iObj < m_constData.size(); ++iObj)
	{
		if (ToBeStopped()) break;

		SetStatus("Writing objects data to result file (object " + std::to_string(iObj + 1) + "/" + std::to_string(m_constData.size()) + ")");

		// write constant data
		if (m_selectors.constProps.id              ) WriteValue(ETXTCommands::OBJECT_ID               , m_constData[iObj].id);
		if (m_selectors.constProps.type            ) WriteValue(ETXTCommands::OBJECT_TYPE             , m_constData[iObj].type);
		if (m_selectors.constProps.geometry        ) WriteValue(ETXTCommands::OBJECT_GEOMETRY         , m_constData[iObj].geometry);
		if (m_selectors.constProps.material        ) WriteValue(ETXTCommands::OBJECT_COMPOUND_TYPE    , m_constData[iObj].compound);
		if (m_selectors.constProps.activityInterval) WriteValue(ETXTCommands::OBJECT_ACTIV_INTERV, m_constData[iObj].activity);

		// store default precision and set selected one for time-dependent data
		const std::streamsize defaultPrecision = m_resFile.precision();
		m_resFile.precision(m_precision);

		// read all time-dependent data for this object from temporary file
		const auto& type = m_constData[iObj].type;
		CByteStream stream;
		stream.Resize(lt[type].entryLen * m_timePoints.size());
		const std::streampos pos = lt[type].startPos + lt[type].entryLen * m_timePoints.size() * counter[type]++;
		m_tdData.binFileR.seekg(pos);
		m_tdData.binFileR.read(&stream.GetDataRef()[0], stream.Size());

		// write time-dependent data
		for (size_t iTime = 0; iTime < m_timePoints.size(); ++iTime)
		{
			if (ToBeStopped()) break;

			// consider activity flags
			if (!m_tdData.tpActive.empty() && !m_tdData.tpActive[m_constData[iObj].id][iTime])
			{
				stream.Ignore(lt[type].entryLen);
				continue;
			}

			WriteValue(ETXTCommands::OBJECT_TIME, m_timePoints[iTime]);
			WriteFromBinData(type, stream);

			m_progress = static_cast<double>(iObj * m_timePoints.size() + iTime) * 100 / (m_timePoints.size() * m_constData.size());
		}
		m_resFile << std::endl;

		// set back default precision
		m_resFile.precision(defaultPrecision);
	}

	// close temporary binary file
	m_tdData.binFileR.close();
}

void CExportAsText::WriteSceneData()
{
	if (m_selectors.sceneInfo.AllOff()) return;

	SetStatus("Writing scene data to result file");

	if (m_selectors.sceneInfo.domain       ) WriteLine(ETXTCommands::SIMULATION_DOMAIN  , m_systemStructure->GetSimulationDomain());
	if (m_selectors.sceneInfo.pbc          ) WriteLine(ETXTCommands::PERIODIC_BOUNDARIES, m_systemStructure->GetPBC().bEnabled, m_systemStructure->GetPBC().bX, m_systemStructure->GetPBC().bY, m_systemStructure->GetPBC().bZ,	m_systemStructure->GetPBC().initDomain);
	if (m_selectors.sceneInfo.anisotropy   ) WriteLine(ETXTCommands::ANISOTROPY         , m_systemStructure->IsAnisotropyEnabled());
	if (m_selectors.sceneInfo.contactRadius) WriteLine(ETXTCommands::CONTACT_RADIUS     , m_systemStructure->IsContactRadiusEnabled());
}

void CExportAsText::WriteGeometriesData()
{
	if (m_selectors.geometries.AllOff()) return;

	SetStatus("Writing geometries data to result file");

	// geometries
	for (const auto* g: m_systemStructure->AllGeometries())
	{
		if (m_selectors.geometries.baseInfo    ) WriteLine(ETXTCommands::GEOMETRY       , g->Name(), g->Key(), g->Mass(), g->FreeMotion());
		if (m_selectors.geometries.tdProperties) WriteLine(ETXTCommands::GEOMETRY_TDVEL , *g->Motion());
		if (m_selectors.geometries.wallsList   ) WriteLine(ETXTCommands::GEOMETRY_PLANES, g->Planes().size(), g->Planes());
	}

	// volumes
	for (const auto* v : m_systemStructure->AllAnalysisVolumes())
		if (m_selectors.geometries.analysisVolumes) WriteLine(ETXTCommands::ANALYSIS_VOLUME, *v);
}

void CExportAsText::WriteMaterialsData()
{
	if (m_selectors.materials.AllOff()) return;

	SetStatus("Writing materials data to result file");

	// compounds
	if (m_selectors.materials.compounds)
	{
		const std::vector<unsigned> compProps = _MUSEN_ACTIVE_PROPERTIES;
		for (const auto* c : m_systemStructure->m_MaterialDatabase.GetCompounds())
		{
			WriteValue(ETXTCommands::MATERIALS_COMPOUNDS, c->GetKey(), c->GetName());
			for (const auto p : compProps)
				WriteValue(p, c->GetProperty(p)->GetValue());
			WriteLine();
		}
	}
	// interactions
	if (m_selectors.materials.interactions)
	{
		const std::vector<unsigned> interProps = _MUSEN_ACTIVE_INTERACTIONS;
		for (const auto* i : m_systemStructure->m_MaterialDatabase.GetInteractions())
		{
			WriteValue(ETXTCommands::MATERIALS_INTERACTIONS, i->GetKey1(), i->GetKey2());
			for (const auto p : interProps)
				WriteValue(p, i->GetProperty(p)->GetValue());
			WriteLine();
		}
	}
	// mixtures
	if (m_selectors.materials.mixtures)
	{
		for (const auto* m : m_systemStructure->m_MaterialDatabase.GetMixtures())
		{
			WriteValue(ETXTCommands::MATERIALS_MIXTURES, m->GetKey(), m->GetName());
			for (size_t iFrac = 0; iFrac < m->FractionsNumber(); ++iFrac)
				WriteValue(iFrac, m->GetFractionCompound(iFrac), m->GetFractionDiameter(iFrac), m->GetFractionContactDiameter(iFrac), m->GetFractionValue(iFrac));
			WriteLine();
		}
	}
}

void CExportAsText::WriteGeneratorsData()
{
	if (m_selectors.generators.AllOff()) return;

	SetStatus("Writing generators data to result file");

	// package generator
	if (m_selectors.generators.packageGenerator)
	{
		for (const auto* g : m_packageGenerator->Generators())
			WriteLine(ETXTCommands::PACKAGE_GENERATOR, *g);
		WriteLine(ETXTCommands::PACKAGE_GENERATOR_CONFIG, *m_packageGenerator);
	}

	// bonds generator
	if (m_selectors.generators.bondsGenerator)
		for (const auto& g : m_bondsGenerator->Generators())
			WriteLine(ETXTCommands::BONDS_GENERATOR, *g);
}

void CExportAsText::TryOpenFileW(std::ofstream& _file, const std::filesystem::path& _name, std::ios::openmode _mode/* = 0*/)
{
	std::filesystem::create_directories(_name.parent_path());
	_file.open(_name, std::ios::out | _mode);
	if (!_file)
	{
		std::string message = "Can not create file: '" + std::filesystem::absolute(_name).string() + "'!";
		if (MUSENFileFunctions::IsDirWriteProtected(_name.parent_path()))
			message += " Path is write protected!";
		RequestErrorStop(message);
	}
}

void CExportAsText::TryOpenFileR(std::ifstream& _file, const std::filesystem::path& _name, std::ios::openmode _mode/* = 0*/)
{
	_file.open(_name, std::ios::in | _mode);
	if (!_file)
		RequestErrorStop("Can not open file: '" + std::filesystem::absolute(_name).string() + "'!");
}

bool CExportAsText::ToBeStopped() const
{
	return m_status == ERunningStatus::TO_BE_STOPPED || m_status == ERunningStatus::FAILED;
}

void CExportAsText::Finalize()
{
	m_resFile.close();
	m_tdData.binFileW.close();
	m_tdData.binFileR.close();
	std::filesystem::remove(m_tdData.binFileName);
	if (m_status == ERunningStatus::IDLE)
		m_progress = 100;
}

void CExportAsText::CalculateBinFileLayout()
{
	auto& lt = m_tdData.binFileLayout; // alias for layout

	/*
	 * Calculates and sets entry length of an object of a given type.
	 */
	const auto CalculateEntryLength = [&](unsigned _type)
	{
		if (lt[_type].objNum)
		{
			const size_t i = VectorFind(m_constData, [&](const SConstantData& _e) { return _e.type == _type; });
			lt[_type].entryLen = GetBinData(i).Size();
		}
	};

	lt.clear();

	// calculate number of specific objects
	for (const auto& obj : m_constData)
		lt[obj.type].objNum++;

	// calculate entry sizes for each object type
	m_systemStructure->PrepareTimePointForRead(m_timePoints.front());
	CalculateEntryLength(SPHERE);
	CalculateEntryLength(SOLID_BOND);
	CalculateEntryLength(TRIANGULAR_WALL);

	// calculate full sizes for each object type
	lt[SPHERE         ].fullLen = lt[SPHERE         ].objNum * lt[SPHERE         ].entryLen * m_timePoints.size();
	lt[SOLID_BOND     ].fullLen = lt[SOLID_BOND     ].objNum * lt[SOLID_BOND     ].entryLen * m_timePoints.size();
	lt[TRIANGULAR_WALL].fullLen = lt[TRIANGULAR_WALL].objNum * lt[TRIANGULAR_WALL].entryLen * m_timePoints.size();

	// calculate starting positions in file for each object type
	lt[SPHERE         ].startPos = 0;
	lt[SOLID_BOND     ].startPos = lt[SPHERE    ].startPos + lt[SPHERE    ].fullLen;
	lt[TRIANGULAR_WALL].startPos = lt[SOLID_BOND].startPos + lt[SOLID_BOND].fullLen;
}

CByteStream CExportAsText::GetBinDataPart(size_t _id) const
{
	CByteStream stream;
	const auto* part = dynamic_cast<const CSphere*>(m_systemStructure->GetObjectByIndex(m_constData[_id].id));
	// NOTE: the sequence of checking flags must be the same when reading from and when writing to the stream
	if (m_selectors.tdPropsPart.angVel      ) stream.Write(part->GetAngleVelocity());
	if (m_selectors.tdPropsPart.coord       ) stream.Write(part->GetCoordinates());
	if (m_selectors.tdPropsPart.force       ) stream.Write(part->GetForce());
	if (m_selectors.tdPropsPart.forceAmpl   ) stream.Write(part->GetForce().Length());
	if (m_selectors.tdPropsPart.orient      ) stream.Write(part->GetOrientation());
	if (m_selectors.tdPropsPart.princStress ) stream.Write(part->GetStressTensor().GetPrincipalStresses());
	if (m_selectors.tdPropsPart.stressTensor) stream.Write(part->GetStressTensor());
	if (m_selectors.tdPropsPart.temperature ) stream.Write(part->GetTemperature());
	if (m_selectors.tdPropsPart.velocity    ) stream.Write(part->GetVelocity());
	return stream;
}

void CExportAsText::WriteFromBinDataPart(CByteStream& _stream)
{
	// NOTE: the sequence of checking flags must be the same when reading from and when writing to the stream
	if (m_selectors.tdPropsPart.angVel      ) WriteValue(ETXTCommands::OBJECT_ANG_VEL      , _stream.Read<CVector3   >());
	if (m_selectors.tdPropsPart.coord       ) WriteValue(ETXTCommands::OBJECT_COORD        , _stream.Read<CVector3   >());
	if (m_selectors.tdPropsPart.force       ) WriteValue(ETXTCommands::OBJECT_FORCE        , _stream.Read<CVector3   >());
	if (m_selectors.tdPropsPart.forceAmpl   ) WriteValue(ETXTCommands::OBJECT_FORCE_AMPL   , _stream.Read<double     >());
	if (m_selectors.tdPropsPart.orient      ) WriteValue(ETXTCommands::OBJECT_ORIENT       , _stream.Read<CQuaternion>());
	if (m_selectors.tdPropsPart.princStress ) WriteValue(ETXTCommands::OBJECT_PRINC_STRESS , _stream.Read<CVector3   >());
	if (m_selectors.tdPropsPart.stressTensor) WriteValue(ETXTCommands::OBJECT_STRESS_TENSOR, _stream.Read<CMatrix3   >());
	if (m_selectors.tdPropsPart.temperature ) WriteValue(ETXTCommands::OBJECT_TEMPERATURE  , _stream.Read<double     >());
	if (m_selectors.tdPropsPart.velocity    ) WriteValue(ETXTCommands::OBJECT_VELOCITY     , _stream.Read<CVector3   >());
}

CByteStream CExportAsText::GetBinDataBond(size_t _id) const
{
	CByteStream stream;
	const auto* bond = dynamic_cast<const CSolidBond*>(m_systemStructure->GetObjectByIndex(m_constData[_id].id));
	// NOTE: the sequence of checking flags must be the same when reading from and when writing to the stream
	if (m_selectors.tdPropsBond.coord      ) stream.Write(m_systemStructure->GetBondCoordinate(m_constData[_id].id));
	if (m_selectors.tdPropsBond.force      ) stream.Write(bond->GetForce());
	if (m_selectors.tdPropsBond.forceAmpl  ) stream.Write(bond->GetForce().Length());
	if (m_selectors.tdPropsBond.tangOverlap) stream.Write(bond->GetTangentialOverlap());
	if (m_selectors.tdPropsBond.temperature) stream.Write(bond->GetTemperature());
	if (m_selectors.tdPropsBond.totTorque  ) stream.Write(bond->GetTotalTorque());
	if (m_selectors.tdPropsBond.velocity   ) stream.Write(m_systemStructure->GetBondVelocity(m_constData[_id].id));
	return stream;
}

void CExportAsText::WriteFromBinDataBond(CByteStream& _stream)
{
	// NOTE: the sequence of checking flags must be the same when reading from and when writing to the stream
	if (m_selectors.tdPropsBond.coord      ) WriteValue(ETXTCommands::OBJECT_COORD       , _stream.Read<CVector3>());
	if (m_selectors.tdPropsBond.force      ) WriteValue(ETXTCommands::OBJECT_FORCE       , _stream.Read<CVector3>());
	if (m_selectors.tdPropsBond.forceAmpl  ) WriteValue(ETXTCommands::OBJECT_FORCE_AMPL  , _stream.Read<double  >());
	if (m_selectors.tdPropsBond.tangOverlap) WriteValue(ETXTCommands::OBJECT_TANG_OVERLAP, _stream.Read<CVector3>());
	if (m_selectors.tdPropsBond.temperature) WriteValue(ETXTCommands::OBJECT_TEMPERATURE , _stream.Read<double  >());
	if (m_selectors.tdPropsBond.totTorque  ) WriteValue(ETXTCommands::OBJECT_TOT_TORQUE  , _stream.Read<double  >());
	if (m_selectors.tdPropsBond.velocity   ) WriteValue(ETXTCommands::OBJECT_VELOCITY    , _stream.Read<CVector3>());
}

CByteStream CExportAsText::GetBinDataWall(size_t _id) const
{
	CByteStream stream;
	const auto* wall = dynamic_cast<const CTriangularWall*>(m_systemStructure->GetObjectByIndex(m_constData[_id].id));
	// NOTE: the sequence of checking flags must be the same when reading from and when writing to the stream
	if (m_selectors.tdPropsWall.coord    ) stream.Write(wall->GetPlaneCoords());
	if (m_selectors.tdPropsWall.force    ) stream.Write(wall->GetForce());
	if (m_selectors.tdPropsWall.forceAmpl) stream.Write(wall->GetForce().Length());
	if (m_selectors.tdPropsWall.velocity ) stream.Write(wall->GetVelocity());
	return stream;
}

void CExportAsText::WriteFromBinDataWall(CByteStream& _stream)
{
	// NOTE: the sequence of checking flags must be the same when reading from and when writing to the stream
	if (m_selectors.tdPropsWall.coord    ) WriteValue(ETXTCommands::OBJECT_PLANE_COORD, _stream.Read<CTriangle>());
	if (m_selectors.tdPropsWall.force    ) WriteValue(ETXTCommands::OBJECT_FORCE      , _stream.Read<CVector3 >());
	if (m_selectors.tdPropsWall.forceAmpl) WriteValue(ETXTCommands::OBJECT_FORCE_AMPL , _stream.Read<double   >());
	if (m_selectors.tdPropsWall.velocity ) WriteValue(ETXTCommands::OBJECT_VELOCITY   , _stream.Read<CVector3 >());
}

CByteStream CExportAsText::GetBinData(size_t _id) const
{
	switch (m_constData[_id].type)
	{
	case SPHERE:          return GetBinDataPart(_id);
	case SOLID_BOND:      return GetBinDataBond(_id);
	case TRIANGULAR_WALL: return GetBinDataWall(_id);
	default:              return CByteStream{};
	}
}

void CExportAsText::WriteFromBinData(unsigned _type, CByteStream& _stream)
{
	switch (_type)
	{
	case SPHERE:          WriteFromBinDataPart(_stream); break;
	case SOLID_BOND:      WriteFromBinDataBond(_stream); break;
	case TRIANGULAR_WALL: WriteFromBinDataWall(_stream); break;
	default: break;
	}
}

template <typename T, typename ... Ts>
void CExportAsText::WriteValue(T&& _val, Ts&&... _vals)
{
	m_resFile << std::forward<T>(_val) << " ";
	((m_resFile << std::forward<Ts>(_vals) << " "), ...);
}

template <typename T, typename ... Ts>
void CExportAsText::WriteValue(ETXTCommands _key, T&& _val, Ts&&... _vals)
{
	m_resFile << E2I(_key) << " ";
	WriteValue(std::forward<T>(_val), std::forward<Ts>(_vals)...);
}

void CExportAsText::WriteLine()
{
	m_resFile << std::endl;
}

template <typename T, typename ... Ts>
void CExportAsText::WriteLine(T&& _val, Ts&&... _vals)
{
	WriteValue(std::forward<T>(_val), std::forward<Ts>(_vals)...);
	WriteLine();
}

template <typename T, typename ... Ts>
void CExportAsText::WriteLine(ETXTCommands _key, T&& _val, Ts&&... _vals)
{
	WriteValue(_key, std::forward<T>(_val), std::forward<Ts>(_vals)...);
	WriteLine();
}