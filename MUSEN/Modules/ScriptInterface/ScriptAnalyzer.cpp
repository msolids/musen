/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ScriptAnalyzer.h"
#include <cctype>

CScriptAnalyzer::CScriptAnalyzer(const std::string& _fileName/* = ""*/)
{
	if (!_fileName.empty())
		AnalyzeFile(_fileName);
}

bool CScriptAnalyzer::AnalyzeFile(const std::string& _fileName)
{
	std::ifstream inputFile;
	inputFile.open(UnicodePath(_fileName)); //open a file
	if (!inputFile.good()) return false;
	std::string line;
	while (safeGetLine(inputFile, line).good())
		ProcessLine(line);
	inputFile.close();
	return true;
}

void CScriptAnalyzer::ProcessLine(const std::string& _line, std::ostream& _out /* = std::cout*/)
{
	// needed for compatibility with older versions
	const auto ReadModelOld = [](std::stringstream& _ss, SJob& _job, EMusenModelType _type)
	{
		const std::string name = GetRestOfLine(&_ss);
		if (!VectorContains(_job.models, [&](const SJob::SModel& _model) { return _model.name == name; }))
			_job.models.push_back(SJob::SModel{ name, "", _type });
	};
	// needed for compatibility with older versions
	const auto ReadModelParamsOld = [](std::stringstream& _ss, SJob& _job, EMusenModelType _type)
	{
		const std::string params = GetRestOfLine(&_ss);
		const size_t i = VectorFind(_job.models, [&](const SJob::SModel& _model) { return _model.type == _type; });
		if (i < _job.models.size())
			_job.models[i].parameters = params;
	};

	if (_line.empty()) return;
	std::stringstream ss{ _line };
	const std::string key = ToUpperCase(GetValueFromStream<std::string>(&ss));
	if (m_jobs.empty() && key != "NEW_JOB" && key != "JOB") // force adding the first job if the first line is not JOB
		m_jobs.push_back({});

	if (key == "NEW_JOB" || key == "JOB")				m_jobs.push_back({});
	else if (key == "SOURCE_FILE")							m_jobs.back().sourceFileName = GetRestOfLine(&ss);
	else if (key == "LOG_FILE")								m_jobs.back().logFileName = GetRestOfLine(&ss);
	else if (key == "RESULT_FILE" || key == "RESULTS_FILE")	m_jobs.back().resultFileName = GetRestOfLine(&ss);
	else if (key == "COMPONENT")
	{
		const std::string value = ToUpperCase(GetValueFromStream<std::string>(&ss));
		if (value.size() == 1 && std::isdigit(value[0])) // to treat old syntax
			m_jobs.back().component = static_cast<SJob::EComponent>(std::stoi(value));
		else
		{
			if (value == "PACKAGE_GENERATOR")	m_jobs.back().component = SJob::EComponent::PACKAGE_GENERATOR;
			else if (value == "RESULTS_ANALYZER")	m_jobs.back().component = SJob::EComponent::RESULTS_ANALYZER;
			else if (value == "BONDS_GENERATOR")	m_jobs.back().component = SJob::EComponent::BONDS_GENERATOR;
			else if (value == "SIMULATOR")			m_jobs.back().component = SJob::EComponent::SIMULATOR;
			else if (value == "SNAPSHOT_GENERATOR")	m_jobs.back().component = SJob::EComponent::SNAPSHOT_GENERATOR;
			else if (value == "EXPORT_TO_TEXT")		m_jobs.back().component = SJob::EComponent::EXPORT_TO_TEXT;
			else if (value == "IMPORT_FROM_TEXT")	m_jobs.back().component = SJob::EComponent::IMPORT_FROM_TEXT;
		}
	}
	else if (key == "AGGLOMERATES_DB")	m_jobs.back().agglomeratesDBFileName = GetRestOfLine(&ss);
	else if (key == "SIMULATOR_TYPE")
	{
		const std::string type = ToUpperCase(GetRestOfLine(&ss));
		if (type == "CPU")																	m_jobs.back().simulatorType = ESimulatorType::CPU;
		if (type == "GPU" || type == "GPU_FAST" || type == "GPU FAST" || type == "GPUFAST") m_jobs.back().simulatorType = ESimulatorType::GPU;
	}
	else if (key == "MODEL_PP")								ReadModelOld(ss, m_jobs.back(), EMusenModelType::PP);
	else if (key == "MODEL_PW")								ReadModelOld(ss, m_jobs.back(), EMusenModelType::PW);
	else if (key == "MODEL_SB")								ReadModelOld(ss, m_jobs.back(), EMusenModelType::SB);
	else if (key == "MODEL_LB")								ReadModelOld(ss, m_jobs.back(), EMusenModelType::LB);
	else if (key == "MODEL_EF" || key == "MODEL_EXT_FORCE")	ReadModelOld(ss, m_jobs.back(), EMusenModelType::EF);
	else if (key == "MODEL_PPHT")							ReadModelOld(ss, m_jobs.back(), EMusenModelType::PP);
	// TODO: adjust documentation and scripts
	else if (key == "MODEL")
	{
		const std::string name = GetRestOfLine(&ss);
		if (!VectorContains(m_jobs.back().models, [&](const SJob::SModel& _model) { return _model.name == name; }))
			m_jobs.back().models.push_back(SJob::SModel{ name, "", EMusenModelType::UNSPECIFIED });
	}
	else if (key == "MODEL_PP_PARAMS")										ReadModelParamsOld(ss, m_jobs.back(), EMusenModelType::PP);
	else if (key == "MODEL_PW_PARAMS")										ReadModelParamsOld(ss, m_jobs.back(), EMusenModelType::PW);
	else if (key == "MODEL_SB_PARAMS")										ReadModelParamsOld(ss, m_jobs.back(), EMusenModelType::SB);
	else if (key == "MODEL_LB_PARAMS")										ReadModelParamsOld(ss, m_jobs.back(), EMusenModelType::LB);
	else if (key == "MODEL_EF_PARAMS" || key == "MODEL_EXT_FORCE_PARAMS")	ReadModelParamsOld(ss, m_jobs.back(), EMusenModelType::EF);
	else if (key == "MODEL_PPHT_PARAMS")									ReadModelParamsOld(ss, m_jobs.back(), EMusenModelType::PP);
	else if (key == "MODEL_PARAMS")
	{
		const auto model = GetValueFromStream<std::string>(&ss);
		const auto params = GetRestOfLine(&ss);
		const size_t i = VectorFind(m_jobs.back().models, [&](const SJob::SModel& _model) { return _model.name == model; });
		if (i != static_cast<size_t>(-1) && i < m_jobs.back().models.size())
			m_jobs.back().models[i].parameters = params;
	}
	else if (key == "SNAPSHOT_TIME")								ss >> m_jobs.back().dSnapshotTP;
	else if (key == "SIMULATION_STEP" || key == "SIMULATION_TSTEP")	ss >> m_jobs.back().dSimulationTimeStep;
	else if (key == "SAVING_STEP" || key == "SAVING_TSTEP")			ss >> m_jobs.back().dSavingTimeStep;
	else if (key == "END_TIME")										ss >> m_jobs.back().dEndSimulationTime;
	else if (key == "SIMULATION_STEP_FACTOR")						ss >> m_jobs.back().simulationStepFactor;
	else if (key == "SAVING_STEP_FACTOR")							ss >> m_jobs.back().savingStepFactor;
	else if (key == "END_TIME_FACTOR")								ss >> m_jobs.back().endTimeFactor;
	else if (key == "SAVE_COLLISIONS")		ss >> m_jobs.back().saveCollsionsFlag;
	else if (key == "CONNECTED_PP_CONTACT")	ss >> m_jobs.back().connectedPPContactFlag;
	else if (key == "ANISOTROPY")			ss >> m_jobs.back().anisotropyFlag;
	else if (key == "DIFF_CONTACT_RADIUS")	ss >> m_jobs.back().contactRadiusFlag;
	else if (key == "EXT_ACCEL")			ss >> m_jobs.back().vExtAccel;
	else if (key == "SIMULATION_DOMAIN")	ss >> m_jobs.back().simulationDomain;
	else if (key == "PBC_FLAGS")            ss >> m_jobs.back().pbcFlags[0] >> m_jobs.back().pbcFlags[1] >> m_jobs.back().pbcFlags[2];
	else if (key == "PBC_DOMAIN")           ss >> m_jobs.back().pbcDomain;
	else if (key == "SELECTIVE_SAVING_P")  { m_jobs.back().selectiveSavingFlag = true; ss >> m_jobs.back().selectiveSavingFlags.bAngVelocity >> m_jobs.back().selectiveSavingFlags.bCoordinates >> m_jobs.back().selectiveSavingFlags.bForce >> m_jobs.back().selectiveSavingFlags.bQuaternion >> m_jobs.back().selectiveSavingFlags.bVelocity >> m_jobs.back().selectiveSavingFlags.bTensor >> m_jobs.back().selectiveSavingFlags.bTemperature; }
	else if (key == "SELECTIVE_SAVING_SB") { m_jobs.back().selectiveSavingFlag = true; ss >> m_jobs.back().selectiveSavingFlags.bSBForce >> m_jobs.back().selectiveSavingFlags.bSBTangOverlap >> m_jobs.back().selectiveSavingFlags.bSBTotTorque; }
	else if (key == "SELECTIVE_SAVING_LB") { m_jobs.back().selectiveSavingFlag = true; ss >> m_jobs.back().selectiveSavingFlags.bLBForce; }
	else if (key == "SELECTIVE_SAVING_TW") { m_jobs.back().selectiveSavingFlag = true; ss >> m_jobs.back().selectiveSavingFlags.bTWPlaneCoord >> m_jobs.back().selectiveSavingFlags.bTWForce >> m_jobs.back().selectiveSavingFlags.bTWVelocity; }
	else if (key == "VERLET_AUTO")			ss >> m_jobs.back().verletAutoFlag;
	else if (key == "VERLET_COEF")			ss >> m_jobs.back().verletCoef;
	else if (key == "VERLET_MAX_CELLS")		ss >> m_jobs.back().iVerletMaxCells;
	else if (key == "VARIABLE_TIME_STEP")	ss >> m_jobs.back().variableTimeStepFlag;
	else if (key == "MAX_PART_MOVE")		ss >> m_jobs.back().maxPartMove;
	else if (key == "STEP_INC_FACTOR")		ss >> m_jobs.back().stepIncFactor;
	else if (key == "STOP_CRITERION")
	{
		const auto criterion = ToUpperCase(GetValueFromStream<std::string>(&ss));
		if		(criterion == "NONE") {
			m_jobs.back().stopCriteria.push_back(CBaseSimulator::EStopCriteria::NONE);
		}
		else if (criterion == "BROKEN_BONDS") {
			m_jobs.back().stopCriteria.push_back(CBaseSimulator::EStopCriteria::BROKEN_BONDS);
			m_jobs.back().stopValues.maxBrokenBonds = GetValueFromStream<size_t>(&ss);
		}
	}
	else if (key == "MONITOR")				m_jobs.back().vMonitors.push_back(GetRestOfLine(&ss));
	else if (key == "POSTPROCESS")			m_jobs.back().vPostProcessCommands.push_back(GetRestOfLine(&ss));
	else if (key.rfind("PACK_GEN", 0) == 0)
	{
		const auto index = GetValueFromStream<size_t>(&ss);
		if      (key == "PACK_GEN_VOLUME")		m_jobs.back().packageGenerators[index].volume     = GetRestOfLine(&ss);
		else if (key == "PACK_GEN_MIXTURE")		m_jobs.back().packageGenerators[index].mixture    = GetRestOfLine(&ss);
		else if (key == "PACK_GEN_POROSITY")	m_jobs.back().packageGenerators[index].porosity   = GetValueFromStream<double>(&ss);
		else if (key == "PACK_GEN_OVERLAP")		m_jobs.back().packageGenerators[index].overlap    = GetValueFromStream<double>(&ss);
		else if (key == "PACK_GEN_ITERATIONS")	m_jobs.back().packageGenerators[index].iterations = static_cast<size_t>(GetValueFromStream<double>(&ss));
		else if (key == "PACK_GEN_VELOCITY")	m_jobs.back().packageGenerators[index].velocity   = GetValueFromStream<CVector3>(&ss);
		else if (key == "PACK_GEN_INSIDE")		m_jobs.back().packageGenerators[index].inside     = GetValueFromStream<CTriState>(&ss);
	}
	else if (key.rfind("BOND_GEN", 0) == 0)
	{
		const auto index = GetValueFromStream<size_t>(&ss);
		if      (key == "BOND_GEN_MATERIAL")	m_jobs.back().bondGenerators[index].material	= GetRestOfLine(&ss);
		else if (key == "BOND_GEN_MINDIST")		m_jobs.back().bondGenerators[index].minDistance	= GetValueFromStream<double>(&ss);
		else if (key == "BOND_GEN_MAXDIST")		m_jobs.back().bondGenerators[index].maxDistance	= GetValueFromStream<double>(&ss);
		else if (key == "BOND_GEN_DIAMETER")	m_jobs.back().bondGenerators[index].diameter	= GetValueFromStream<double>(&ss);
		else if (key == "BOND_GEN_OVERLAY")		m_jobs.back().bondGenerators[index].overlay		= GetValueFromStream<CTriState>(&ss);
	}
	else if (key == "MATERIAL_PROPERTY")
	{
		const auto propertyStr = ToUpperCase(GetValueFromStream<std::string>(&ss));
		const auto compoundKey = GetValueFromStream<std::string>(&ss);
		const auto value = GetValueFromStream<double>(&ss);
		ETPPropertyTypes propertyKey{ PROPERTY_NO_PROPERTY };
		if		(propertyStr == "DENSITY")				propertyKey = PROPERTY_DENSITY;
		else if (propertyStr == "DYNAMIC_VISCOSITY")	propertyKey = PROPERTY_DYNAMIC_VISCOSITY;
		else if (propertyStr == "YOUNG_MODULUS")		propertyKey = PROPERTY_YOUNG_MODULUS;
		else if (propertyStr == "NORMAL_STRENGTH")		propertyKey = PROPERTY_NORMAL_STRENGTH;
		else if (propertyStr == "TANGENTIAL_STRENGTH")	propertyKey = PROPERTY_TANGENTIAL_STRENGTH;
		else if (propertyStr == "POISSON_RATIO")		propertyKey = PROPERTY_POISSON_RATIO;
		else if (propertyStr == "SURFACE_ENERGY")		propertyKey = PROPERTY_SURFACE_ENERGY;
		else if (propertyStr == "ATOMIC_VOLUME")		propertyKey = PROPERTY_ATOMIC_VOLUME;
		else if (propertyStr == "SURFACE_TENSION")		propertyKey = PROPERTY_SURFACE_TENSION;
		else if (propertyStr == "TIME_THERM_EXP_COEFF") propertyKey = PROPERTY_TIME_THERM_EXP_COEFF;
		else if (propertyStr == "YIELD_STRENGTH")		propertyKey = PROPERTY_YIELD_STRENGTH;
		else _out << "Unknown material property: " << propertyStr << std::endl;
		if (propertyKey != PROPERTY_NO_PROPERTY)
			m_jobs.back().materialProperties.push_back(SJob::SMDBMaterialProperties{ propertyKey, compoundKey, value });
	}
	else if (key == "INTERACTION_PROPERTY")
	{
		const auto propertyStr = ToUpperCase(GetValueFromStream<std::string>(&ss));
		const auto compoundKey1 = GetValueFromStream<std::string>(&ss);
		const auto compoundKey2 = GetValueFromStream<std::string>(&ss);
		const auto value = GetValueFromStream<double>(&ss);
		EIntPropertyTypes propertyKey{ PROPERTY_INT_NO_PROPERTY };
		if		(propertyStr == "RESTITUTION_COEFFICIENT")	propertyKey = PROPERTY_RESTITUTION_COEFFICIENT;
		else if (propertyStr == "SLIDING_FRICTION")			propertyKey = PROPERTY_STATIC_FRICTION;
		else if (propertyStr == "ROLLING_FRICTION")			propertyKey = PROPERTY_ROLLING_FRICTION;
		else _out << "Unknown interaction property: " << propertyStr << std::endl;
		if (propertyKey != PROPERTY_INT_NO_PROPERTY)
			m_jobs.back().interactionProperties.push_back(SJob::SMDBInteractionProperties{ propertyKey, compoundKey1, compoundKey2, value });
	}
	else if (key == "MIXTURE_PROPERTY")
	{
		const auto iMixture        = GetValueFromStream<size_t>(&ss);
		const auto iFraction       = GetValueFromStream<size_t>(&ss);
		const auto compoundKey     = GetValueFromStream<std::string>(&ss);
		const auto diameter        = GetValueFromStream<double>(&ss);
		const auto contactDiameter = GetValueFromStream<double>(&ss);
		const auto fraction        = GetValueFromStream<double>(&ss);
		m_jobs.back().mixtureProperties.push_back(SJob::SMDBMixtureProperties{ iMixture - 1, iFraction - 1, compoundKey, diameter, contactDiameter, fraction });
	}
	else if (key == "TEXT_EXPORT_OBJECTS")
	{
		m_jobs.back().txtExportSettings.objectTypes.particles = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.objectTypes.bonds     = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.objectTypes.walls     = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_CONST")
	{
		m_jobs.back().txtExportSettings.constProps.id               = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.constProps.type             = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.constProps.geometry         = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.constProps.material         = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.constProps.activityInterval = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_TD_PART")
	{
		m_jobs.back().txtExportSettings.tdPropsPart.angVel      = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.coord       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.force       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.forceAmpl   = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.orient      = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.princStress = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.stressTensor= GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.temperature = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsPart.velocity    = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_TD_BOND")
	{
		m_jobs.back().txtExportSettings.tdPropsBond.coord       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsBond.force       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsBond.forceAmpl   = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsBond.tangOverlap = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsBond.temperature = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsBond.totTorque   = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsBond.velocity    = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_TD_WALL")
	{
		m_jobs.back().txtExportSettings.tdPropsWall.coord     = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsWall.force     = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsWall.forceAmpl = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.tdPropsWall.velocity  = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_SCENE")
	{
		m_jobs.back().txtExportSettings.sceneInfo.domain        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.sceneInfo.pbc           = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.sceneInfo.anisotropy    = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.sceneInfo.contactRadius = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_GEOMETRIES")
	{
		m_jobs.back().txtExportSettings.geometries.baseInfo        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.geometries.tdProperties    = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.geometries.wallsList       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.geometries.analysisVolumes = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_MATERIALS")
	{
		m_jobs.back().txtExportSettings.materials.compounds    = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.materials.interactions = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.materials.mixtures     = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_GENERATORS")
	{
		m_jobs.back().txtExportSettings.generators.packageGenerator = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportSettings.generators.bondsGenerator   = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_PRECISION")
	{
		m_jobs.back().txtPrecision = GetValueFromStream<int>(&ss);
	}
	else if (key == "TIME_INTERVAL")
	{
		m_jobs.back().timeBeg = GetValueFromStream<double>(&ss);
		m_jobs.back().timeEnd = GetValueFromStream<double>(&ss);
	}
	else if (key == "GEOMETRY_MOTION_TIME")
	{
		SJob::SGeometryMotionIntervalTime motion;
		const std::string nameOrIndex = GetValueFromStream<std::string>(&ss);
		motion.geometryName                      = IsSimpleUInt(nameOrIndex) ? "" : nameOrIndex;
		motion.geometryIndex                     = IsSimpleUInt(nameOrIndex) ? std::stoull(nameOrIndex) : -1;
		motion.intrerval.timeBeg                 = GetValueFromStream<double>(&ss);
		motion.intrerval.timeEnd                 = GetValueFromStream<double>(&ss);
		motion.intrerval.motion.velocity         = GetValueFromStream<CVector3>(&ss);
		motion.intrerval.motion.rotationVelocity = GetValueFromStream<CVector3>(&ss);
		motion.intrerval.motion.rotationCenter   = GetValueFromStream<CVector3>(&ss);
		m_jobs.back().geometryTimeIntervals.push_back(motion);
	}
	else if (key == "GEOMETRY_MOTION_FORCE")
	{
		SJob::SGeometryMotionIntervalForce motion;
		const std::string nameOrIndex            = GetValueFromStream<std::string>(&ss);
		motion.geometryName                      = IsSimpleUInt(nameOrIndex) ? "" : nameOrIndex;
		motion.geometryIndex                     = IsSimpleUInt(nameOrIndex) ? std::stoull(nameOrIndex) : -1;
		motion.intrerval.forceLimit              = GetValueFromStream<double>(&ss);
		const std::string type = ToUpperCase(GetValueFromStream<std::string>(&ss));
		motion.intrerval.limitType               = type == "MIN" ? CGeometryMotion::SForceMotionInterval::ELimitType::MIN : CGeometryMotion::SForceMotionInterval::ELimitType::MAX;
		motion.intrerval.motion.velocity         = GetValueFromStream<CVector3>(&ss);
		motion.intrerval.motion.rotationVelocity = GetValueFromStream<CVector3>(&ss);
		motion.intrerval.motion.rotationCenter   = GetValueFromStream<CVector3>(&ss);
		m_jobs.back().geometryForceIntervals.push_back(motion);
	}
	else
		_out << "Unknown script key: " << key << std::endl;
}

size_t CScriptAnalyzer::JobsCount() const
{
	return m_jobs.size();
}

std::vector<SJob> CScriptAnalyzer::Jobs() const
{
	return m_jobs;
}
