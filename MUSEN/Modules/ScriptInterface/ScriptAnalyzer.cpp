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
	if (_line.empty()) return;
	std::stringstream ss{ _line };
	const std::string key = ToUpperCase(GetValueFromStream<std::string>(&ss));
	if (m_jobs.empty() && key != "NEW_JOB" && key != "JOB") // force adding the first job if the first line is not JOB
		m_jobs.push_back({});

	if      (key == "NEW_JOB" || key == "JOB")				m_jobs.push_back({});
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
			if      (value == "PACKAGE_GENERATOR")	m_jobs.back().component = SJob::EComponent::PACKAGE_GENERATOR;
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
	else if (key == "MODEL_PP")								m_jobs.back().models[EMusenModelType::PP].name = GetRestOfLine(&ss);
	else if (key == "MODEL_PW")								m_jobs.back().models[EMusenModelType::PW].name = GetRestOfLine(&ss);
	else if (key == "MODEL_SB")								m_jobs.back().models[EMusenModelType::SB].name = GetRestOfLine(&ss);
	else if (key == "MODEL_LB")								m_jobs.back().models[EMusenModelType::LB].name = GetRestOfLine(&ss);
	else if (key == "MODEL_EF" || key == "MODEL_EXT_FORCE")	m_jobs.back().models[EMusenModelType::EF].name = GetRestOfLine(&ss);
	else if (key == "MODEL_PP_PARAMS")										m_jobs.back().models[EMusenModelType::PP].parameters = GetRestOfLine(&ss);
	else if (key == "MODEL_PW_PARAMS")										m_jobs.back().models[EMusenModelType::PW].parameters = GetRestOfLine(&ss);
	else if (key == "MODEL_SB_PARAMS")										m_jobs.back().models[EMusenModelType::SB].parameters = GetRestOfLine(&ss);
	else if (key == "MODEL_LB_PARAMS")										m_jobs.back().models[EMusenModelType::LB].parameters = GetRestOfLine(&ss);
	else if (key == "MODEL_EF_PARAMS" || key == "MODEL_EXT_FORCE_PARAMS")	m_jobs.back().models[EMusenModelType::EF].parameters = GetRestOfLine(&ss);
	else if (key == "SNAPSHOT_TIME")								ss >> m_jobs.back().dSnapshotTP;
	else if (key == "SIMULATION_STEP" || key == "SIMULATION_TSTEP")	ss >> m_jobs.back().dSimulationTimeStep;
	else if (key == "SAVING_STEP" || key == "SAVING_TSTEP")			ss >> m_jobs.back().dSavingTimeStep;
	else if (key == "END_TIME")										ss >> m_jobs.back().dEndSimulationTime;
	else if (key == "SAVE_COLLISIONS")		ss >> m_jobs.back().saveCollsionsFlag;
	else if (key == "CONNECTED_PP_CONTACT")	ss >> m_jobs.back().connectedPPContactFlag;
	else if (key == "ANISOTROPY")			ss >> m_jobs.back().anisotropyFlag;
	else if (key == "DIFF_CONTACT_RADIUS")	ss >> m_jobs.back().contactRadiusFlag;
	else if (key == "EXT_ACCEL")			ss >> m_jobs.back().vExtAccel;
	else if (key == "SIMULATION_DOMAIN")	ss >> m_jobs.back().simulationDomain;
	else if (key == "SELECTIVE_SAVING_P")  { m_jobs.back().selectiveSavingFlag = true; ss >> m_jobs.back().selectiveSavingFlags.bAngVelocity >> m_jobs.back().selectiveSavingFlags.bCoordinates >> m_jobs.back().selectiveSavingFlags.bForce >> m_jobs.back().selectiveSavingFlags.bQuaternion >> m_jobs.back().selectiveSavingFlags.bVelocity >> m_jobs.back().selectiveSavingFlags.bTensor; }
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
		const auto iMixture    = GetValueFromStream<size_t>(&ss);
		const auto iFraction   = GetValueFromStream<size_t>(&ss);
		const auto compoundKey = GetValueFromStream<std::string>(&ss);
		const auto diameter    = GetValueFromStream<double>(&ss);
		const auto fraction    = GetValueFromStream<double>(&ss);
		m_jobs.back().mixtureProperties.push_back(SJob::SMDBMixtureProperties{ iMixture - 1, iFraction - 1, compoundKey, diameter, fraction });
	}
	else if (key == "TEXT_EXPORT_OBJECTS")
	{
		m_jobs.back().txtExportObjects.particles       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportObjects.solidBonds      = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportObjects.liquidBonds     = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportObjects.triangularWalls = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_SCENE")
	{
		m_jobs.back().txtExportScene.domain        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportScene.pbc           = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportScene.anisotropy    = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportScene.contactRadius = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_CONST")
	{
		m_jobs.back().txtExportConst.id               = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportConst.type             = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportConst.geometry         = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportConst.material         = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportConst.activityInterval = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_TD")
	{
		m_jobs.back().txtExportTD.coordinate        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.velocity          = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.angularVelocity   = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.totalForce        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.force             = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.quaternion        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.stressTensor      = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.totalTorque       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.tangOverlap       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportTD.temperature       = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_GEOMETRIES")
	{
		m_jobs.back().txtExportGeometries.baseInfo        = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportGeometries.tdProperties    = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportGeometries.wallsList       = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportGeometries.analysisVolumes = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_MATERIALS")
	{
		m_jobs.back().txtExportMaterials.compounds    = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportMaterials.interactions = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportMaterials.mixtures     = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_GENERATORS")
	{
		m_jobs.back().txtExportGenerators.packageGenerator = GetValueFromStream<CTriState>(&ss).ToBool(true);
		m_jobs.back().txtExportGenerators.bondsGenerator   = GetValueFromStream<CTriState>(&ss).ToBool(true);
	}
	else if (key == "TEXT_EXPORT_PRECISION")
	{
		m_jobs.back().txtPrecision = GetValueFromStream<int>(&ss);
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
