/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BaseSimulator.h"

CBaseSimulator::CBaseSimulator() : m_VerletList(m_Scene)
{
	m_simulatorType = ESimulatorType::BASE;

	m_dCurrentTime = 0;
	m_dEndTime = DEFAULT_END_TIME;
	m_initSimulationStep = DEFAULT_SIMULATION_STEP;
	m_currSimulationStep = DEFAULT_SIMULATION_STEP;
	m_dSavingStep = DEFAULT_SAVING_STEP;
	m_dLastSavingTime = 0;

	m_bPredictionStep = true;
	m_nCellsMax = DEFAULT_MAX_CELLS;
	m_dVerletDistanceCoeff = DEFAULT_VERLET_DISTANCE_COEFF;
	m_bAutoAdjustVerletDistance = true;
	m_bWallsVelocityChanged = true;
	m_bConsiderAnisotropy = false;
	m_bVariableTimeStep = false;
	m_bVariableTimeStep = false;

	m_nInactiveParticlesNumber = 0;
	m_nBrockenBondsNumber = 0;
	m_nBrockenLiquidBondsNumber = 0;
	m_nGeneratedObjects = 0;

	m_dMaxParticleVelocity = 0;
	m_dMaxWallVelocity = 0;
	m_partMoveLimit = 1e-8;
	m_timeStepFactor = 1.01;

	m_nCurrentStatus = ERunningStatus::IDLE;
	m_vecExternalAccel = CVector3{ 0, 0, -GRAVITY_CONSTANT };

	m_pModelManager = nullptr;
	m_pGenerationManager = nullptr;

	m_pPPModel = nullptr;
	m_pPWModel = nullptr;
	m_pSBModel = nullptr;
	m_pLBModel = nullptr;
	m_pEFModel = nullptr;

	// selective saving
	m_bSelectiveSaving = false;
	m_SSelectiveSavingFlags.SetAll(true);
}

CBaseSimulator::CBaseSimulator(const CBaseSimulator& _simulator) : m_VerletList(m_Scene)
{
	m_dMaxParticleVelocity = 0;
	m_dMaxWallVelocity = 0;

	p_CopySimulatorData(_simulator);
}

CBaseSimulator& CBaseSimulator::operator=(const CBaseSimulator& _simulator)
{
	p_CopySimulatorData(_simulator);
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
	SetExternalAccel(ProtoVectorToVector(sim.external_acceleration()));
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
	m_bSelectiveSaving = m_pSystemStructure->GetSimulationInfo()->selective_saving();
	if (m_pSystemStructure->GetSimulationInfo()->has_selective_saving_flags())
	{
		const ProtoSelectiveSaving& selectiveSaving = m_pSystemStructure->GetSimulationInfo()->selective_saving_flags();
		// particles
		m_SSelectiveSavingFlags.bCoordinates   = selectiveSaving.p_coord();
		m_SSelectiveSavingFlags.bVelocity      = selectiveSaving.p_vel();
		m_SSelectiveSavingFlags.bAngVelocity   = selectiveSaving.p_angvel();
		m_SSelectiveSavingFlags.bQuaternion    = selectiveSaving.p_quatern();
		m_SSelectiveSavingFlags.bForce         = selectiveSaving.p_force();
		m_SSelectiveSavingFlags.bTensor        = selectiveSaving.p_tensor();
		// solid bonds
		m_SSelectiveSavingFlags.bSBForce       = selectiveSaving.sb_force();
		m_SSelectiveSavingFlags.bSBTangOverlap = selectiveSaving.sb_tangoverlap();
		m_SSelectiveSavingFlags.bSBTotTorque   = selectiveSaving.sb_tottorque();
		// liquid bonds
		m_SSelectiveSavingFlags.bLBForce       = selectiveSaving.lb_force();
		// triangular walls
		m_SSelectiveSavingFlags.bTWPlaneCoord  = selectiveSaving.tw_coord();
		m_SSelectiveSavingFlags.bTWForce       = selectiveSaving.tw_force();
		m_SSelectiveSavingFlags.bTWVelocity    = selectiveSaving.tw_vel();
	}
	else
		m_SSelectiveSavingFlags.SetAll(true);

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
	pSim->set_end_time(m_dEndTime);
	pSim->set_simulation_step(m_initSimulationStep);
	pSim->set_saving_step(m_dSavingStep);
	VectorToProtoVector(pSim->mutable_external_acceleration(), m_vecExternalAccel);
	pSim->set_simulator_type(static_cast<unsigned>(m_simulatorType));
	pSim->set_max_cells_number(m_nCellsMax);
	pSim->set_verlet_dist_coeff(m_dVerletDistanceCoeff);
	pSim->set_verlet_auto_adjust(m_bAutoAdjustVerletDistance);
	pSim->set_flexible_time_step(m_bVariableTimeStep);
	pSim->set_part_move_limit(m_partMoveLimit);
	pSim->set_time_step_factor(m_timeStepFactor);

	// save selective saving parameters
	m_pSystemStructure->GetSimulationInfo()->set_selective_saving(m_bSelectiveSaving);
	ProtoSelectiveSaving* selectiveSaving = m_pSystemStructure->GetSimulationInfo()->mutable_selective_saving_flags();
	// particles
	selectiveSaving->set_p_coord(m_SSelectiveSavingFlags.bCoordinates);
	selectiveSaving->set_p_vel(m_SSelectiveSavingFlags.bVelocity);
	selectiveSaving->set_p_angvel(m_SSelectiveSavingFlags.bAngVelocity);
	selectiveSaving->set_p_quatern(m_SSelectiveSavingFlags.bQuaternion);
	selectiveSaving->set_p_force(m_SSelectiveSavingFlags.bForce);
	selectiveSaving->set_p_tensor(m_SSelectiveSavingFlags.bTensor);
	// solid bonds
	selectiveSaving->set_sb_force(m_SSelectiveSavingFlags.bSBForce);
	selectiveSaving->set_sb_tangoverlap(m_SSelectiveSavingFlags.bSBTangOverlap);
	selectiveSaving->set_sb_tottorque(m_SSelectiveSavingFlags.bSBTotTorque);
	// liquid bonds
	selectiveSaving->set_lb_force(m_SSelectiveSavingFlags.bLBForce);
	// triangular walls
	selectiveSaving->set_tw_coord(m_SSelectiveSavingFlags.bTWPlaneCoord);
	selectiveSaving->set_tw_force(m_SSelectiveSavingFlags.bTWForce);
	selectiveSaving->set_tw_vel(m_SSelectiveSavingFlags.bTWVelocity);

	// save additional stop criteria
	auto* protoSC = m_pSystemStructure->GetSimulationInfo()->mutable_stop_criteria();
	protoSC->clear_types();
	for (const auto criterion : m_stopCriteria)
		protoSC->add_types(E2I(criterion));
	protoSC->set_max_broken_bonds(m_stopValues.maxBrokenBonds);
}

bool CBaseSimulator::GetVariableTimeStep() const
{
	return m_bVariableTimeStep;
}

void CBaseSimulator::SetVariableTimeStep(bool _bFlag)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_bVariableTimeStep = _bFlag;
	if (m_nCurrentStatus == ERunningStatus::PAUSED)
		m_currSimulationStep = m_initSimulationStep;
}

double CBaseSimulator::GetPartMoveLimit() const
{
	return m_partMoveLimit;
}

void CBaseSimulator::SetPartMoveLimit(double _dx)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_partMoveLimit = _dx;
}

double CBaseSimulator::GetTimeStepFactor() const
{
	return m_timeStepFactor;
}

void CBaseSimulator::SetTimeStepFactor(double _factor)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_timeStepFactor = _factor;
}

bool CBaseSimulator::IsSelectiveSavingEnabled() const
{
	return m_bSelectiveSaving;
}

SSelectiveSavingFlags CBaseSimulator::GetSelectiveSavingFlags() const
{
	return m_SSelectiveSavingFlags;
}

const CModelManager* CBaseSimulator::GetModelManager() const
{
	return m_pModelManager;
}

CModelManager* CBaseSimulator::GetModelManager()
{
	return m_pModelManager;
}

void CBaseSimulator::SetModelManager(CModelManager* _pModelManager)
{
	m_pModelManager = _pModelManager;
}

const CGenerationManager* CBaseSimulator::GetGenerationManager() const
{
	return m_pGenerationManager;
}

void CBaseSimulator::SetGenerationManager(CGenerationManager* _pGenerationManager)
{
	m_pGenerationManager = _pGenerationManager;
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
	return m_nCurrentStatus;
}

void CBaseSimulator::SetCurrentStatus(const ERunningStatus& _nNewStatus)
{
	m_nCurrentStatus = _nNewStatus;
}

double CBaseSimulator::GetCurrentTime() const
{
	return m_dCurrentTime;
}

void CBaseSimulator::SetCurrentTime(double _dTime)
{
	m_dCurrentTime = _dTime;
}

double CBaseSimulator::GetEndTime() const
{
	return m_dEndTime;
}

double CBaseSimulator::GetCurrSimulationStep() const
{
	return m_currSimulationStep;
}

double CBaseSimulator::GetSavingStep() const
{
	return m_dSavingStep;
}

uint32_t CBaseSimulator::GetMaxCells() const
{
	return m_nCellsMax;
}

double CBaseSimulator::GetVerletCoeff() const
{
	return m_dVerletDistanceCoeff;
}

bool CBaseSimulator::GetAutoAdjustFlag() const
{
	return m_bAutoAdjustVerletDistance;
}

size_t CBaseSimulator::GetNumberOfInactiveParticles() const
{
	return m_nInactiveParticlesNumber;
}

size_t CBaseSimulator::GetNumberOfBrockenBonds() const
{
	return m_nBrockenBondsNumber;
}

size_t CBaseSimulator::GetNumberOfBrockenLiquidBonds() const
{
	return m_nBrockenLiquidBondsNumber;
}

size_t CBaseSimulator::GetNumberOfGeneratedObjects() const
{
	return m_nGeneratedObjects;
}

double CBaseSimulator::GetMaxParticleVelocity() const
{
	return m_dMaxParticleVelocity;
}

void CBaseSimulator::SetEndTime(double _dEndTime)
{
	if ((m_nCurrentStatus != ERunningStatus::IDLE) && (m_nCurrentStatus != ERunningStatus::PAUSED)) return;
	m_dEndTime = _dEndTime < 0 ? DEFAULT_END_TIME : _dEndTime;
}

double CBaseSimulator::GetInitSimulationStep() const
{
	return m_initSimulationStep;
}

void CBaseSimulator::SetInitSimulationStep(double _timeStep)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_initSimulationStep = _timeStep <= 0 ? DEFAULT_SIMULATION_STEP : _timeStep;
}

void CBaseSimulator::SetCurrSimulationStep(double _timeStep)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_currSimulationStep = _timeStep <= 0 ? DEFAULT_SIMULATION_STEP : _timeStep;
}

void CBaseSimulator::SetSavingStep(double _dSavingStep)
{
	if ((m_nCurrentStatus != ERunningStatus::IDLE) && (m_nCurrentStatus != ERunningStatus::PAUSED)) return;
	m_dSavingStep = _dSavingStep <= 0 ? DEFAULT_SAVING_STEP : _dSavingStep;
}

void CBaseSimulator::SetMaxCells(uint32_t _nMaxCells)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_nCellsMax = _nMaxCells;
}

void CBaseSimulator::SetVerletCoeff(double _dCoeff)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_dVerletDistanceCoeff = _dCoeff;
}

void CBaseSimulator::SetAutoAdjustFlag(bool _bFlag)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	m_bAutoAdjustVerletDistance = _bFlag;
}

CVector3 CBaseSimulator::GetExternalAccel() const
{
	return m_vecExternalAccel;
}

void CBaseSimulator::SetExternalAccel(const CVector3& _accel)
{
	m_vecExternalAccel = _accel;
}

std::string CBaseSimulator::IsDataCorrect() const
{
	// check that exists all materials of all objects in the database
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
	if ( !m_pSystemStructure->IsAllCompoundsDefined().empty())
		return m_pSystemStructure->IsAllCompoundsDefined();
	return m_pGenerationManager->IsDataCorrect();
}

void CBaseSimulator::p_CopySimulatorData(const CBaseSimulator& _simulator)
{
	SetEndTime(_simulator.m_dEndTime);
	SetInitSimulationStep(_simulator.m_initSimulationStep);
	SetCurrSimulationStep(_simulator.m_currSimulationStep);
	SetSavingStep(_simulator.m_dSavingStep);
	SetMaxCells(_simulator.m_nCellsMax);
	SetVerletCoeff(_simulator.m_dVerletDistanceCoeff);
	SetAutoAdjustFlag(_simulator.m_bAutoAdjustVerletDistance);
	SetSelectiveSaving(_simulator.m_bSelectiveSaving);
	SetSelectiveSavingParameters(_simulator.m_SSelectiveSavingFlags);
	SetVariableTimeStep(_simulator.m_bVariableTimeStep);
	SetPartMoveLimit(_simulator.m_partMoveLimit);
	SetTimeStepFactor(_simulator.m_timeStepFactor);

	m_nInactiveParticlesNumber = _simulator.m_nInactiveParticlesNumber;
	m_nBrockenBondsNumber = _simulator.m_nBrockenBondsNumber;
	m_nBrockenLiquidBondsNumber = _simulator.m_nBrockenLiquidBondsNumber;
	m_nGeneratedObjects = _simulator.m_nGeneratedObjects;

	SetSystemStructure(_simulator.m_pSystemStructure);
	SetCurrentStatus(_simulator.m_nCurrentStatus);
	SetExternalAccel(_simulator.m_vecExternalAccel);

	SetModelManager(_simulator.m_pModelManager);
	SetGenerationManager(_simulator.m_pGenerationManager);

	m_pPPModel = _simulator.m_pPPModel;
	m_pPWModel = _simulator.m_pPWModel;
	m_pSBModel = _simulator.m_pSBModel;
	m_pLBModel = _simulator.m_pLBModel;
	m_pEFModel = _simulator.m_pEFModel;

	m_stopCriteria = _simulator.m_stopCriteria;
	m_stopValues = _simulator.m_stopValues;
}

void CBaseSimulator::p_InitializeModels()
{
	m_pPPModel = dynamic_cast<CParticleParticleModel*>(m_pModelManager->GetModel(EMusenModelType::PP));
	m_pPWModel = dynamic_cast<CParticleWallModel*>(m_pModelManager->GetModel(EMusenModelType::PW));
	m_pSBModel = dynamic_cast<CSolidBondModel*>(m_pModelManager->GetModel(EMusenModelType::SB));
	m_pLBModel = dynamic_cast<CLiquidBondModel*>(m_pModelManager->GetModel(EMusenModelType::LB));
	m_pEFModel = dynamic_cast<CExternalForceModel*>(m_pModelManager->GetModel(EMusenModelType::EF));

	m_models.clear();
	if (m_pPPModel) m_models.push_back(m_pPPModel);
	if (m_pPWModel) m_models.push_back(m_pPWModel);
	if (m_pSBModel) m_models.push_back(m_pSBModel);
	if (m_pLBModel) m_models.push_back(m_pLBModel);
	if (m_pEFModel) m_models.push_back(m_pEFModel);

	for (auto& model : m_models)
		model->SetPBC(m_Scene.GetPBC());

	m_VerletList.SetConnectedPPContact(m_pModelManager->GetConnectedPPContact());
}

size_t CBaseSimulator::p_GenerateNewObjects()
{
	size_t nNewParticles = m_pGenerationManager->GenerateObjects(m_dCurrentTime, m_Scene);
	if (nNewParticles > 0)
	{
		m_VerletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_Scene.GetMinParticleContactRadius(), m_Scene.GetMaxParticleContactRadius(),
			m_nCellsMax, m_dVerletDistanceCoeff, m_bAutoAdjustVerletDistance);
		m_VerletList.ResetCurrentData();
		m_Scene.InitializeMaterials();
		m_nGeneratedObjects += nNewParticles;
		m_Scene.UpdateParticlesToBonds();
	}
	return nNewParticles;
}

void CBaseSimulator::p_SaveData()
{
	if (m_bSelectiveSaving)
		m_pSystemStructure->GetSimulationInfo()->set_selective_saving(true);
	else
		m_pSystemStructure->GetSimulationInfo()->set_selective_saving(false);

	if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bTensor)
	{
		m_vAddSavingDataPart.resize(m_Scene.GetTotalParticlesNumber());
		PrepareAdditionalSavingData();
	}

	m_pSystemStructure->PrepareTimePointForWrite(m_dCurrentTime);

	// save particles properties
	SParticleStruct& particles = m_Scene.GetRefToParticles();
	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		CPhysicalObject* pPart = m_pSystemStructure->GetObjectByIndex(particles.InitIndex(i));
		if (!particles.Active(i) && !pPart->IsActive(m_dCurrentTime)) return;
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bCoordinates) pPart->SetCoordinates(m_Scene.GetObjectCoord(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bVelocity)    pPart->SetVelocity(m_Scene.GetObjectVel(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bAngVelocity) pPart->SetAngleVelocity(m_Scene.GetObjectAnglVel(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bForce)       pPart->SetForce(particles.Force(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bQuaternion)  pPart->SetOrientation(particles.Quaternion(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bTensor)      pPart->SetStressTensor(m_vAddSavingDataPart[i].sStressTensor);
		pPart->SetObjectActivity(particles.Active(i) ? m_dCurrentTime : particles.EndActivity(i), particles.Active(i));
	});

	// save solid bonds properties
	SSolidBondStruct& solidBonds = m_Scene.GetRefToSolidBonds();
	ParallelFor(m_Scene.GetBondsNumber(), [&](size_t i)
	{
		auto* pSBond = dynamic_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex(solidBonds.InitIndex(i)));
		if (!solidBonds.Active(i) && !pSBond->IsActive(m_dCurrentTime)) return;
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bSBForce)       pSBond->SetForce(solidBonds.TotalForce(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bSBTangOverlap) pSBond->SetTangentialOverlap(solidBonds.TangentialOverlap(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bSBTotTorque)   pSBond->SetTotalTorque(Length(solidBonds.NormalMoment(i) + solidBonds.TangentialMoment(i)));
		pSBond->SetObjectActivity(solidBonds.Active(i) ? m_dCurrentTime : solidBonds.EndActivity(i), solidBonds.Active(i));
	});

	// save liquid bonds properties
	//std::vector<std::unique_ptr<SLiquidBondStruct>>& liquidBonds = m_Scene.GetRefToLiquidBonds();
	ParallelFor(m_Scene.GetLiquidBondsNumber(), [&](size_t i)
	{
		SLiquidBondStruct& liqbond = m_Scene.GetRefToLiquidBonds();
		CPhysicalObject* pLBond = m_pSystemStructure->GetObjectByIndex(liqbond.InitIndex(i));
		if (!liqbond.Active(i) && !pLBond->IsActive(m_dCurrentTime)) return;
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bLBForce) pLBond->SetForce(liqbond.NormalForce(i) + liqbond.TangentialForce(i));
		pLBond->SetObjectActivity(liqbond.Active(i) ? m_dCurrentTime : liqbond.EndActivity(i), liqbond.Active(i));
	});

	// save wall properties
	SWallStruct& walls = m_Scene.GetRefToWalls();
	ParallelFor(walls.Size(), [&](size_t i)
	{
		auto* pWall = dynamic_cast<CTriangularWall*>(m_pSystemStructure->GetObjectByIndex(walls.InitIndex(i)));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bTWPlaneCoord) pWall->SetPlaneCoord(walls.Vert1(i), walls.Vert2(i), walls.Vert3(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bTWForce)      pWall->SetForce(walls.Force(i));
		if (!m_bSelectiveSaving || m_SSelectiveSavingFlags.bTWVelocity)   pWall->SetVelocity(walls.Vel(i));
	});

	for (const auto& function : m_additionalSavingSteps)
		function();
}

void CBaseSimulator::PrintStatus() const
{
	using namespace std::chrono;

	// calculate progress
	const double currProgress = m_dCurrentTime / m_dEndTime;
	const int64_t currDuration = duration_cast<milliseconds>(steady_clock::now() - m_chronoSimStart).count() - m_chronoPauseLength;
	const int64_t remainingTime = static_cast<int64_t>(currDuration / currProgress) - currDuration;
	std::time_t endTime = system_clock::to_time_t(system_clock::now() + milliseconds(remainingTime));

	// print out status and progress
	*p_out << "Current time [s]: " << m_dCurrentTime << std::endl;
	*p_out << "\tMax particle velocity [m/s]:  " << m_dMaxParticleVelocity << std::endl;
	*p_out << "\tCurrent progress:             " << Double2Percent(currProgress) << std::endl;
	*p_out << "\tTime left [d:h:m:s]:          " << MsToTimeSpan(remainingTime) << std::endl;
	*p_out << "\tWill finish at [d.m.y h:m:s]: " << std::put_time(std::localtime(&endTime), "%d.%m.%y %H:%M:%S") << std::endl;
}

void CBaseSimulator::SetSelectiveSaving(bool _bSelectiveSaving)
{
	m_bSelectiveSaving = _bSelectiveSaving;
}

void CBaseSimulator::SetSelectiveSavingParameters(const SSelectiveSavingFlags& _SSelectiveSavingFlags)
{
	m_SSelectiveSavingFlags = _SSelectiveSavingFlags;
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

bool CBaseSimulator::AdditionalStopCriterionMet()
{
	if (m_stopCriteria.empty()) return false;

	for (const auto& criterion : m_stopCriteria)
	{
		switch (criterion)
		{
		case EStopCriteria::NONE: break;
		case EStopCriteria::BROKEN_BONDS:
			if (m_nBrockenBondsNumber > m_stopValues.maxBrokenBonds)
				return true;
			break;
		}
	}

	return false;
}
