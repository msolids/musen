/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BaseSimulator.h"

CBaseSimulator::CBaseSimulator(const CBaseSimulator& _other) :
	CMusenComponent{ _other }
{
	CopySimulatorData(_other);
}

CBaseSimulator& CBaseSimulator::operator=(const CBaseSimulator& _simulator)
{
	if (&_simulator != this)
		CopySimulatorData(_simulator);
	return *this;
}

void CBaseSimulator::LoadConfiguration()
{
	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!protoMessage.has_simulator()) return;
	const ProtoModuleSimulator& sim = protoMessage.simulator();
	SetEndTime(sim.end_time());
	SetInitSimulationStep(sim.simulation_step());
	SetSavingStep(sim.saving_step());
	SetExternalAccel(Proto2Val(sim.external_acceleration()));
	SetType(static_cast<ESimulatorType>(sim.simulator_type()));
	if (sim.max_cells_number())
	{
		SetMaxCells(sim.max_cells_number());
		SetVerletCoeff(sim.verlet_dist_coeff());
		SetAutoAdjustFlag(sim.verlet_auto_adjust());
	}
	SetVariableTimeStep(sim.flexible_time_step());
	SetPartMoveLimit(sim.part_move_limit());
	SetTimeStepFactor(sim.time_step_factor());

	// load selective saving parameters
	m_selectiveSaving = m_pSystemStructure->GetSimulationInfo()->selective_saving();
	if (m_pSystemStructure->GetSimulationInfo()->has_selective_saving_flags())
	{
		const ProtoSelectiveSaving& selectiveSaving = m_pSystemStructure->GetSimulationInfo()->selective_saving_flags();
		// particles
		m_selectiveSavingFlags.bCoordinates   = selectiveSaving.p_coord();
		m_selectiveSavingFlags.bVelocity      = selectiveSaving.p_vel();
		m_selectiveSavingFlags.bAngVelocity   = selectiveSaving.p_angvel();
		m_selectiveSavingFlags.bQuaternion    = selectiveSaving.p_quatern();
		m_selectiveSavingFlags.bForce         = selectiveSaving.p_force();
		m_selectiveSavingFlags.bTensor        = selectiveSaving.p_tensor();
		m_selectiveSavingFlags.bTemperature =	selectiveSaving.p_temperature();
		// solid bonds
		m_selectiveSavingFlags.bSBForce       = selectiveSaving.sb_force();
		m_selectiveSavingFlags.bSBTangOverlap = selectiveSaving.sb_tangoverlap();
		m_selectiveSavingFlags.bSBTotTorque   = selectiveSaving.sb_tottorque();
		// liquid bonds
		m_selectiveSavingFlags.bLBForce       = selectiveSaving.lb_force();
		// triangular walls
		m_selectiveSavingFlags.bTWPlaneCoord  = selectiveSaving.tw_coord();
		m_selectiveSavingFlags.bTWForce       = selectiveSaving.tw_force();
		m_selectiveSavingFlags.bTWVelocity    = selectiveSaving.tw_vel();
	}
	else
		m_selectiveSavingFlags.SetAll(true);

	// load additional stop criteria
	const auto& protoSC = m_pSystemStructure->GetSimulationInfo()->stop_criteria();
	m_stopCriteria.clear();
	for (const auto criterion : protoSC.types())
		m_stopCriteria.push_back(static_cast<EStopCriteria>(criterion));
	m_stopValues.maxBrokenBonds = protoSC.max_broken_bonds();
}

void CBaseSimulator::SaveConfiguration()
{
	ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	ProtoModuleSimulator* pSim = protoMessage.mutable_simulator();
	pSim->set_end_time(m_endTime);
	pSim->set_simulation_step(m_initSimulationStep);
	pSim->set_saving_step(m_savingStep);
	Val2Proto(pSim->mutable_external_acceleration(), m_externalAcceleration);
	pSim->set_simulator_type(static_cast<unsigned>(m_simulatorType));
	pSim->set_max_cells_number(m_cellsMax);
	pSim->set_verlet_dist_coeff(m_verletDistanceCoeff);
	pSim->set_verlet_auto_adjust(m_autoAdjustVerletDistance);
	pSim->set_flexible_time_step(m_variableTimeStep);
	pSim->set_part_move_limit(m_partMoveLimit);
	pSim->set_time_step_factor(m_timeStepFactor);

	// save selective saving parameters
	m_pSystemStructure->GetSimulationInfo()->set_selective_saving(m_selectiveSaving);
	ProtoSelectiveSaving* selectiveSaving = m_pSystemStructure->GetSimulationInfo()->mutable_selective_saving_flags();
	// particles
	selectiveSaving->set_p_coord(m_selectiveSavingFlags.bCoordinates);
	selectiveSaving->set_p_vel(m_selectiveSavingFlags.bVelocity);
	selectiveSaving->set_p_angvel(m_selectiveSavingFlags.bAngVelocity);
	selectiveSaving->set_p_quatern(m_selectiveSavingFlags.bQuaternion);
	selectiveSaving->set_p_force(m_selectiveSavingFlags.bForce);
	selectiveSaving->set_p_tensor(m_selectiveSavingFlags.bTensor);
	selectiveSaving->set_p_temperature(m_selectiveSavingFlags.bTemperature);
	// solid bonds
	selectiveSaving->set_sb_force(m_selectiveSavingFlags.bSBForce);
	selectiveSaving->set_sb_tangoverlap(m_selectiveSavingFlags.bSBTangOverlap);
	selectiveSaving->set_sb_tottorque(m_selectiveSavingFlags.bSBTotTorque);
	// liquid bonds
	selectiveSaving->set_lb_force(m_selectiveSavingFlags.bLBForce);
	// triangular walls
	selectiveSaving->set_tw_coord(m_selectiveSavingFlags.bTWPlaneCoord);
	selectiveSaving->set_tw_force(m_selectiveSavingFlags.bTWForce);
	selectiveSaving->set_tw_vel(m_selectiveSavingFlags.bTWVelocity);

	// save additional stop criteria
	auto* protoSC = m_pSystemStructure->GetSimulationInfo()->mutable_stop_criteria();
	protoSC->clear_types();
	for (const auto criterion : m_stopCriteria)
		protoSC->add_types(E2I(criterion));
	protoSC->set_max_broken_bonds(m_stopValues.maxBrokenBonds);
}

bool CBaseSimulator::GetVariableTimeStep() const
{
	return m_variableTimeStep;
}

void CBaseSimulator::SetVariableTimeStep(bool _bFlag)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_variableTimeStep = _bFlag;
	if (m_status == ERunningStatus::PAUSED)
		m_currSimulationStep = m_initSimulationStep;
}

double CBaseSimulator::GetPartMoveLimit() const
{
	return m_partMoveLimit;
}

void CBaseSimulator::SetPartMoveLimit(double _dx)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_partMoveLimit = _dx;
}

double CBaseSimulator::GetTimeStepFactor() const
{
	return m_timeStepFactor;
}

void CBaseSimulator::SetTimeStepFactor(double _factor)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_timeStepFactor = _factor;
}

bool CBaseSimulator::IsSelectiveSavingEnabled() const
{
	return m_selectiveSaving;
}

SSelectiveSavingFlags CBaseSimulator::GetSelectiveSavingFlags() const
{
	return m_selectiveSavingFlags;
}

const CModelManager* CBaseSimulator::GetModelManager() const
{
	return m_modelManager;
}

CModelManager* CBaseSimulator::GetModelManager()
{
	return m_modelManager;
}

void CBaseSimulator::SetModelManager(CModelManager* _pModelManager)
{
	m_modelManager = _pModelManager;
}

const CGenerationManager* CBaseSimulator::GetGenerationManager() const
{
	return m_generationManager;
}

void CBaseSimulator::SetGenerationManager(CGenerationManager* _pGenerationManager)
{
	m_generationManager = _pGenerationManager;
}

const CSystemStructure* CBaseSimulator::GetSystemStructure() const
{
	return m_pSystemStructure;
}

ESimulatorType CBaseSimulator::GetType() const
{
	return m_simulatorType;
}

void CBaseSimulator::SetType(const ESimulatorType& _type)
{
	m_simulatorType = _type;
}

ERunningStatus CBaseSimulator::GetCurrentStatus() const
{
	return m_status;
}

void CBaseSimulator::SetCurrentStatus(const ERunningStatus& _nNewStatus)
{
	m_status = _nNewStatus;
}

std::chrono::time_point<std::chrono::system_clock> CBaseSimulator::GetStartDateTime() const
{
	return m_chronoSimStart;
}

std::chrono::time_point<std::chrono::system_clock> CBaseSimulator::GetFinishDateTime() const
{
	const double currProgress = m_currentTime / m_endTime;
	const int64_t currDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_chronoSimStart).count() - m_chronoPauseLength;
	const int64_t remainingTime = static_cast<int64_t>(static_cast<double>(currDuration) / currProgress) - currDuration;
	return std::chrono::system_clock::now() + std::chrono::milliseconds(remainingTime);
}

double CBaseSimulator::GetCurrentTime() const
{
	return m_currentTime;
}

void CBaseSimulator::SetCurrentTime(double _dTime)
{
	m_currentTime = _dTime;
}

double CBaseSimulator::GetEndTime() const
{
	return m_endTime;
}

double CBaseSimulator::GetCurrSimulationStep() const
{
	return m_currSimulationStep;
}

double CBaseSimulator::GetSavingStep() const
{
	return m_savingStep;
}

uint32_t CBaseSimulator::GetMaxCells() const
{
	return m_cellsMax;
}

double CBaseSimulator::GetVerletCoeff() const
{
	return m_verletDistanceCoeff;
}

bool CBaseSimulator::GetAutoAdjustFlag() const
{
	return m_autoAdjustVerletDistance;
}

size_t CBaseSimulator::GetNumberOfInactiveParticles() const
{
	return m_nInactiveParticles;
}

size_t CBaseSimulator::GetNumberOfBrokenBonds() const
{
	return m_nBrokenBonds;
}

size_t CBaseSimulator::GetNumberOfBrokenLiquidBonds() const
{
	return m_nBrokenLiquidBonds;
}

size_t CBaseSimulator::GetNumberOfGeneratedObjects() const
{
	return m_nGeneratedObjects;
}

double CBaseSimulator::GetMaxParticleVelocity() const
{
	return m_maxParticleVelocity;
}

void CBaseSimulator::SetEndTime(double _dEndTime)
{
	if ((m_status != ERunningStatus::IDLE) && (m_status != ERunningStatus::PAUSED)) return;
	m_endTime = _dEndTime < 0 ? DEFAULT_END_TIME : _dEndTime;
}

double CBaseSimulator::GetInitSimulationStep() const
{
	return m_initSimulationStep;
}

void CBaseSimulator::SetInitSimulationStep(double _timeStep)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_initSimulationStep = _timeStep <= 0 ? DEFAULT_SIMULATION_STEP : _timeStep;
}

void CBaseSimulator::SetCurrSimulationStep(double _timeStep)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_currSimulationStep = _timeStep <= 0 ? DEFAULT_SIMULATION_STEP : _timeStep;
}

void CBaseSimulator::SetSavingStep(double _dSavingStep)
{
	if ((m_status != ERunningStatus::IDLE) && (m_status != ERunningStatus::PAUSED)) return;
	m_savingStep = _dSavingStep <= 0 ? DEFAULT_SAVING_STEP : _dSavingStep;
}

void CBaseSimulator::SetMaxCells(uint32_t _nMaxCells)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_cellsMax = _nMaxCells;
}

void CBaseSimulator::SetVerletCoeff(double _dCoeff)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_verletDistanceCoeff = _dCoeff;
}

void CBaseSimulator::SetAutoAdjustFlag(bool _bFlag)
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;
	m_autoAdjustVerletDistance = _bFlag;
}

CVector3 CBaseSimulator::GetExternalAccel() const
{
	return m_externalAcceleration;
}

void CBaseSimulator::SetExternalAccel(const CVector3& _accel)
{
	m_externalAcceleration = _accel;
}

std::string CBaseSimulator::IsDataCorrect() const
{
	// check that exists all materials of all objects in the database
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
	if ( !m_pSystemStructure->IsAllCompoundsDefined().empty())
		return m_pSystemStructure->IsAllCompoundsDefined();
	return m_generationManager->IsDataCorrect();
}

size_t CBaseSimulator::GenerateNewObjects()
{
	const size_t nNewParticles = m_generationManager->GenerateObjects(m_currentTime, m_scene);
	if (nNewParticles > 0)
	{
		m_verletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_scene.GetMinParticleContactRadius(), m_scene.GetMaxParticleContactRadius(),
			m_cellsMax, m_verletDistanceCoeff, m_autoAdjustVerletDistance);
		m_verletList.ResetCurrentData();
		m_scene.InitializeMaterials();
		m_nGeneratedObjects += nNewParticles;
		m_scene.UpdateParticlesToBonds();
	}
	return nNewParticles;
}

void CBaseSimulator::p_SaveData()
{
	if (m_selectiveSaving)
		m_pSystemStructure->GetSimulationInfo()->set_selective_saving(true);
	else
		m_pSystemStructure->GetSimulationInfo()->set_selective_saving(false);

	if (!m_selectiveSaving || m_selectiveSavingFlags.bTensor)
	{
		m_additionalSavingData.resize(m_scene.GetTotalParticlesNumber());
		PrepareAdditionalSavingData();
	}

	m_pSystemStructure->PrepareTimePointForWrite(m_currentTime);

	// save particles properties
	SParticleStruct& particles = m_scene.GetRefToParticles();
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		CPhysicalObject* pPart = m_pSystemStructure->GetObjectByIndex(particles.InitIndex(i));
		if (!particles.Active(i) && !pPart->IsActive(m_currentTime)) return;
		if (!m_selectiveSaving || m_selectiveSavingFlags.bCoordinates) pPart->SetCoordinates(m_scene.GetObjectCoord(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bVelocity)    pPart->SetVelocity(m_scene.GetObjectVel(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bAngVelocity) pPart->SetAngleVelocity(m_scene.GetObjectAnglVel(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bForce)       pPart->SetForce(particles.Force(i));
		if ((!m_selectiveSaving || m_selectiveSavingFlags.bQuaternion) && particles.QuaternionExist()) pPart->SetOrientation(particles.Quaternion(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bTensor)      pPart->SetStressTensor(m_additionalSavingData[i].stressTensor);
		if ((!m_selectiveSaving || m_selectiveSavingFlags.bTemperature) && particles.ThermalsExist()) pPart->SetTemperature(particles.Temperature(i));

		pPart->SetObjectActivity(particles.Active(i) ? m_currentTime : particles.EndActivity(i), particles.Active(i));
	});

	// save solid bonds properties
	SSolidBondStruct& solidBonds = m_scene.GetRefToSolidBonds();
	ParallelFor(m_scene.GetBondsNumber(), [&](size_t i)
	{
		auto* pSBond = dynamic_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex(solidBonds.InitIndex(i)));
		if (!solidBonds.Active(i) && !pSBond->IsActive(m_currentTime)) return;
		if (!m_selectiveSaving || m_selectiveSavingFlags.bSBForce)       pSBond->SetForce(solidBonds.TotalForce(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bSBTangOverlap) pSBond->SetTangentialOverlap(solidBonds.TangentialOverlap(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bSBTotTorque)   pSBond->SetTotalTorque(Length(solidBonds.NormalMoment(i) + solidBonds.TangentialMoment(i)));
		pSBond->SetObjectActivity(solidBonds.Active(i) ? m_currentTime : solidBonds.EndActivity(i), solidBonds.Active(i));
	});

	// save liquid bonds properties
	SLiquidBondStruct& liqbond = m_scene.GetRefToLiquidBonds();
	ParallelFor(m_scene.GetLiquidBondsNumber(), [&](size_t i)
	{
		CPhysicalObject* pLBond = m_pSystemStructure->GetObjectByIndex(liqbond.InitIndex(i));
		if (!liqbond.Active(i) && !pLBond->IsActive(m_currentTime)) return;
		if (!m_selectiveSaving || m_selectiveSavingFlags.bLBForce) pLBond->SetForce(liqbond.NormalForce(i) + liqbond.TangentialForce(i));
		pLBond->SetObjectActivity(liqbond.Active(i) ? m_currentTime : liqbond.EndActivity(i), liqbond.Active(i));
	});

	// save wall properties
	SWallStruct& walls = m_scene.GetRefToWalls();
	ParallelFor(walls.Size(), [&](size_t i)
	{
		auto* pWall = dynamic_cast<CTriangularWall*>(m_pSystemStructure->GetObjectByIndex(walls.InitIndex(i)));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bTWPlaneCoord) pWall->SetPlaneCoord(walls.Vert1(i), walls.Vert2(i), walls.Vert3(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bTWForce)      pWall->SetForce(walls.Force(i));
		if (!m_selectiveSaving || m_selectiveSavingFlags.bTWVelocity)   pWall->SetVelocity(walls.Vel(i));
	});

	for (const auto& function : m_additionalSavingSteps)
		function();
}

void CBaseSimulator::PrintStatus() const
{
	using namespace std::chrono;

	// calculate progress
	const double currProgress = m_currentTime / m_endTime;
	const auto finishTimePoint = GetFinishDateTime();
	const auto finishTimeTimePointC = system_clock::to_time_t(finishTimePoint);
	const int64_t remainingMs = duration_cast<milliseconds>(finishTimePoint - system_clock::now()).count();

	// print out status and progress
	*p_out << "Current time [s]: " << m_currentTime << std::endl;
	*p_out << "\tMax particle velocity [m/s]:  " << m_maxParticleVelocity << std::endl;
	*p_out << "\tMax particle temperature [K]: " << m_maxParticleTemperature << std::endl;
	*p_out << "\tCurrent progress:             " << Double2Percent(currProgress) << std::endl;
	*p_out << "\tTime left [d:h:m:s]:          " << MsToTimeSpan(remainingMs) << std::endl;
	*p_out << "\tWill finish at [d.m.y h:m:s]: " << std::put_time(std::localtime(&finishTimeTimePointC), "%d.%m.%y %H:%M:%S") << std::endl;
}

void CBaseSimulator::MoveObjectsStep(double _timeStep, bool _predictionStep)
{
	MoveParticles(_predictionStep);
	MoveWalls(_timeStep);
}

void CBaseSimulator::SetSelectiveSaving(bool _bSelectiveSaving)
{
	m_selectiveSaving = _bSelectiveSaving;
}

void CBaseSimulator::SetSelectiveSavingParameters(const SSelectiveSavingFlags& _SSelectiveSavingFlags)
{
	m_selectiveSavingFlags = _SSelectiveSavingFlags;
}

std::list<std::function<void()>>::iterator CBaseSimulator::AddSavingStep(const std::function<void()>& _function)
{
	if (!_function) return {};
	m_additionalSavingSteps.push_back(_function);
	return m_additionalSavingSteps.end();
}

std::vector<CBaseSimulator::EStopCriteria> CBaseSimulator::GetStopCriteria() const
{
	return m_stopCriteria;
}

CBaseSimulator::SStopValues CBaseSimulator::GetStopValues() const
{
	return m_stopValues;
}

void CBaseSimulator::SetStopCriteria(const std::vector<EStopCriteria>& _criteria)
{
	m_stopCriteria = _criteria;
}

void CBaseSimulator::SetStopValues(const SStopValues& _values)
{
	m_stopValues = _values;
}

void CBaseSimulator::Initialize()
{
	// time parameters
	m_currentTime = 0;
	m_lastSavingTime = m_currentTime;
	m_isPredictionStep = true;
	m_currSimulationStep = m_initSimulationStep;

	// time-dependent values
	m_wallsVelocityChanged = true;

	// statistics
	m_nInactiveParticles = 0;
	m_nBrokenBonds = 0;
	m_nBrokenLiquidBonds = 0;
	m_nGeneratedObjects = 0;

	// settings
	m_considerAnisotropy = m_pSystemStructure->IsAnisotropyEnabled();

	// delete old data
	m_pSystemStructure->ClearAllStatesFrom(m_currentTime);

	// scene
	m_optionalSceneVars = GetModelManager()->GetUtilizedVariables();
	m_scene.SetSystemStructure(m_pSystemStructure);
	m_scene.InitializeScene(m_currentTime, m_optionalSceneVars);

	// generation manager
	m_generationManager->Initialize();

	// verlet list
	m_verletList.InitializeList();
	m_verletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_scene.GetMinParticleContactRadius(), m_scene.GetMaxParticleContactRadius(), m_cellsMax, m_verletDistanceCoeff, m_autoAdjustVerletDistance);
	m_verletList.ResetCurrentData();

	// models
	InitializeModels();
}

void CBaseSimulator::InitializeModels()
{
	m_PPModels.clear();
	m_PWModels.clear();
	m_SBModels.clear();
	m_LBModels.clear();
	m_EFModels.clear();

	m_models = m_modelManager->GetAllActiveModels();

	for (auto* model : m_models)
	{
		if      (dynamic_cast<CParticleParticleModel*>(model)) m_PPModels.push_back(dynamic_cast<CParticleParticleModel*>(model));
		else if (dynamic_cast<CParticleWallModel    *>(model)) m_PWModels.push_back(dynamic_cast<CParticleWallModel    *>(model));
		else if (dynamic_cast<CSolidBondModel       *>(model)) m_SBModels.push_back(dynamic_cast<CSolidBondModel       *>(model));
		else if (dynamic_cast<CLiquidBondModel      *>(model)) m_LBModels.push_back(dynamic_cast<CLiquidBondModel      *>(model));
		else if (dynamic_cast<CExternalForceModel   *>(model)) m_EFModels.push_back(dynamic_cast<CExternalForceModel   *>(model));
	}

	m_verletList.SetConnectedPPContact(m_modelManager->GetConnectedPPContact());

	InitializeModelParameters();
}

void CBaseSimulator::InitializeModelParameters()
{
	for (auto& model : m_models)
		model->SetPBC(m_scene.GetPBC());
}

void CBaseSimulator::Simulate()
{
	if (m_status != ERunningStatus::IDLE && m_status != ERunningStatus::PAUSED) return;

	// performance measurement
	if (m_status == ERunningStatus::IDLE)
	{
		m_chronoPauseLength = 0;
		m_chronoSimStart = std::chrono::system_clock::now();
	}
	else if (m_status == ERunningStatus::PAUSED)
		m_chronoPauseLength += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_chronoPauseStart).count();

	RandomSeed(); // needed for some contact models

	if (m_status == ERunningStatus::IDLE)
		Initialize();

	// start simulation
	m_status = ERunningStatus::RUNNING;
	StartSimulation();

	// performance measurement
	if (m_status == ERunningStatus::TO_BE_PAUSED)
		m_chronoPauseStart = std::chrono::system_clock::now();
	else if (m_status == ERunningStatus::TO_BE_STOPPED)
	{
		// output performance statistic
		const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_chronoSimStart).count();
		*p_out << "Elapsed time [d:h:m:s]: " << MsToTimeSpan(elapsedTime) << std::endl;
	}

	// set status
	if (m_status == ERunningStatus::TO_BE_STOPPED)
		m_status = ERunningStatus::IDLE;
	else if (m_status == ERunningStatus::TO_BE_PAUSED)
		m_status = ERunningStatus::PAUSED;
}

void CBaseSimulator::StartSimulation()
{
	while (m_status != ERunningStatus::TO_BE_STOPPED && m_status != ERunningStatus::TO_BE_PAUSED)
	{
		PreCalculationStep();
		UpdateCollisionsStep(m_currSimulationStep);
		CalculateForcesStep(m_currSimulationStep);
		MoveObjectsStep(m_currSimulationStep, m_isPredictionStep);
		if (m_optionalSceneVars.bThermals)
			UpdateTemperatures(m_isPredictionStep);

		if (m_isPredictionStep) // makes prediction step
			m_isPredictionStep = false;
		else
			m_currentTime += m_currSimulationStep;

		// analyze about possible end of interval
		if (m_currentTime >= m_endTime || g_extSignal != 0)
			m_status = ERunningStatus::TO_BE_STOPPED;
	}

	if (m_status == ERunningStatus::TO_BE_STOPPED)
		FinalizeSimulation();
}

void CBaseSimulator::FinalizeSimulation()
{
	if (!g_extSignal)	// if stopped by signal, only close file properly; do not save current time point, as this does not coincide with savings step
		SaveData();
	m_pSystemStructure->SaveToFile();
}

void CBaseSimulator::PreCalculationStep()
{
	const size_t newObjects = GenerateNewObjects();

	if (std::fabs(m_currentTime - m_lastSavingTime) + 0.1 * m_currSimulationStep > m_savingStep || newObjects)
	{
		SaveData();
		m_lastSavingTime = m_currentTime;
		PrintStatus();
		if (AdditionalStopCriterionMet())
			m_status = ERunningStatus::TO_BE_STOPPED;
	}

	if (m_scene.m_PBC.bEnabled)
		UpdatePBC();
}

bool CBaseSimulator::AdditionalStopCriterionMet()
{
	if (m_stopCriteria.empty()) return false;

	for (const auto& criterion : m_stopCriteria)
	{
		switch (criterion)
		{
		case EStopCriteria::NONE: break;
		case EStopCriteria::BROKEN_BONDS:
			if (m_nBrokenBonds > m_stopValues.maxBrokenBonds)
				return true;
			break;
		}
	}

	return false;
}

void CBaseSimulator::CopySimulatorData(const CBaseSimulator& _other)
{
	SetEndTime(_other.m_endTime);
	SetInitSimulationStep(_other.m_initSimulationStep);
	SetCurrSimulationStep(_other.m_currSimulationStep);
	SetSavingStep(_other.m_savingStep);
	SetMaxCells(_other.m_cellsMax);
	SetVerletCoeff(_other.m_verletDistanceCoeff);
	SetAutoAdjustFlag(_other.m_autoAdjustVerletDistance);
	SetSelectiveSaving(_other.m_selectiveSaving);
	SetSelectiveSavingParameters(_other.m_selectiveSavingFlags);
	SetVariableTimeStep(_other.m_variableTimeStep);
	SetPartMoveLimit(_other.m_partMoveLimit);
	SetTimeStepFactor(_other.m_timeStepFactor);

	m_nInactiveParticles = _other.m_nInactiveParticles;
	m_nBrokenBonds = _other.m_nBrokenBonds;
	m_nBrokenLiquidBonds = _other.m_nBrokenLiquidBonds;
	m_nGeneratedObjects = _other.m_nGeneratedObjects;

	SetSystemStructure(_other.m_pSystemStructure);
	SetCurrentStatus(_other.m_status);
	SetExternalAccel(_other.m_externalAcceleration);

	SetModelManager(_other.m_modelManager);
	SetGenerationManager(_other.m_generationManager);

	m_PPModels   = _other.m_PPModels;
	m_PWModels   = _other.m_PWModels;
	m_SBModels   = _other.m_SBModels;
	m_LBModels   = _other.m_LBModels;
	m_EFModels   = _other.m_EFModels;

	m_stopCriteria = _other.m_stopCriteria;
	m_stopValues = _other.m_stopValues;
}

void CBaseSimulator::SAdditionalSavingData::AddStress(const CVector3& _contVector, const CVector3& _force, double _volume)
{
	stressTensor.values[0][0] += _contVector.x * _force.x / _volume;
	stressTensor.values[0][1] += _contVector.x * _force.y / _volume;
	stressTensor.values[0][2] += _contVector.x * _force.z / _volume;

	stressTensor.values[1][0] += _contVector.y * _force.x / _volume;
	stressTensor.values[1][1] += _contVector.y * _force.y / _volume;
	stressTensor.values[1][2] += _contVector.y * _force.z / _volume;

	stressTensor.values[2][0] += _contVector.z * _force.x / _volume;
	stressTensor.values[2][1] += _contVector.z * _force.y / _volume;
	stressTensor.values[2][2] += _contVector.z * _force.z / _volume;
}