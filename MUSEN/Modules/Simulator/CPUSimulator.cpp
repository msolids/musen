/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CPUSimulator.h"

CCPUSimulator::CCPUSimulator() : m_CollCalculator(m_Scene)
{
	InitializeSimulator();
}

CCPUSimulator::CCPUSimulator(const CBaseSimulator& _simulator) : m_CollCalculator(m_Scene)
{
	p_CopySimulatorData(_simulator);
	InitializeSimulator();
}

void CCPUSimulator::InitializeSimulator()
{
	m_bAnalyzeCollisions = false;
	m_CollCalculator.SetPointers(&m_VerletList, &m_CollisionsAnalyzer);

	m_nThreadsNumber = GetThreadsNumber();
	m_vTempCollPPArray.resize(m_nThreadsNumber);
	m_vTempCollPWArray.resize(m_nThreadsNumber);
	for (size_t i = 0; i < m_vTempCollPPArray.size(); i++)
	{
		m_vTempCollPPArray[i].resize(m_nThreadsNumber);
		m_vTempCollPWArray[i].resize(m_nThreadsNumber);
	}
}

void CCPUSimulator::LoadConfiguration()
{
	CBaseSimulator::LoadConfiguration();

	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!protoMessage.has_simulator()) return;
	const ProtoModuleSimulator& sim = protoMessage.simulator();
	EnableCollisionsAnalysis(sim.save_collisions());
}

void CCPUSimulator::SaveConfiguration()
{
	CBaseSimulator::SaveConfiguration();

	ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	ProtoModuleSimulator* pSim = protoMessage.mutable_simulator();
	pSim->set_save_collisions(m_bAnalyzeCollisions);
}

void CCPUSimulator::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	m_CollCalculator.SetSystemStructure(_pSystemStructure);
	m_CollisionsAnalyzer.SetSystemStructure(_pSystemStructure);
	m_pSystemStructure = _pSystemStructure;
}

void CCPUSimulator::EnableCollisionsAnalysis(bool _bEnable)
{
	if (m_nCurrentStatus != ERunningStatus::IDLE) return;

	m_bAnalyzeCollisions = _bEnable;
	m_CollCalculator.EnableCollisionsAnalysis(m_bAnalyzeCollisions);
}

bool CCPUSimulator::IsCollisionsAnalysisEnabled() const
{
	return m_bAnalyzeCollisions;
}

void CCPUSimulator::GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID)
{
	_dMaxOverlap = 0;
	_dAverageOverlap = 0;
	size_t nCollNumber = 0;
	for (const auto& collisions : m_CollCalculator.m_vCollMatrixPP)
		for (const auto coll : collisions)
		{
			if ((coll->nSrcID < _nMaxParticleID) || (coll->nDstID < _nMaxParticleID))
			{
				_dMaxOverlap = std::max(_dMaxOverlap, coll->dNormalOverlap);
				_dAverageOverlap += coll->dNormalOverlap;
				nCollNumber++;
			}
		}

	const SParticleStruct& particles = m_Scene.GetRefToParticles();
	for (const auto& collisions : m_CollCalculator.m_vCollMatrixPW)
		for (const auto coll : collisions)
		{
			if (coll->nDstID < _nMaxParticleID)
			{
				const CVector3 vRc = _VIRTUAL_COORDINATE(particles.Coord(coll->nDstID), coll->nVirtShift, m_Scene.m_PBC) - coll->vContactVector;
				const double dOverlap = particles.ContactRadius(coll->nDstID) - vRc.Length();
				_dMaxOverlap = std::max(_dMaxOverlap, dOverlap);
				_dAverageOverlap += dOverlap;
				nCollNumber++;
			}
		}

	if (nCollNumber)
		_dAverageOverlap = _dAverageOverlap / nCollNumber;
}

void CCPUSimulator::StartSimulation()
{
	if (m_nCurrentStatus != ERunningStatus::IDLE && m_nCurrentStatus != ERunningStatus::PAUSED) return;
	RandomSeed(); // needed for some contact models

	if (m_nCurrentStatus == ERunningStatus::IDLE)
	{
		// set initial time
		m_dCurrentTime = 0; // in the future it should be startTime
		m_dLastSavingTime = m_dCurrentTime;
		m_bPredictionStep = true;
		m_currSimulationStep = m_initSimulationStep;

		m_nInactiveParticlesNumber = 0;
		m_nBrockenBondsNumber = 0;
		m_nBrockenLiquidBondsNumber = 0;
		m_nGeneratedObjects = 0;

		// get flag of anisotropy
		m_bConsiderAnisotropy = m_pSystemStructure->IsAnisotropyEnabled();

		// delete old data
		m_pSystemStructure->ClearAllStatesFrom(m_dCurrentTime);

		// delete all collisions data
		m_CollCalculator.ClearCollMatrixes();
		m_CollCalculator.ClearFinishedCollisionMatrixes();
		if (m_bAnalyzeCollisions)
			m_CollisionsAnalyzer.ResetAndClear();

		m_Scene.SetSystemStructure(m_pSystemStructure);
		m_Scene.InitializeScene(m_dCurrentTime, GetModelManager()->GetUtilizedVariables());
		m_Scene.SaveVerletCoords();
		m_pGenerationManager->Initialize();

		InitializeModels();

		// initialize verlet list
		m_VerletList.InitializeList();
		m_VerletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_Scene.GetMinParticleContactRadius(), m_Scene.GetMaxParticleContactRadius(),
								m_nCellsMax, m_dVerletDistanceCoeff, m_bAutoAdjustVerletDistance);
		m_VerletList.ResetCurrentData();
	}

	// performance measurement
	if (m_nCurrentStatus == ERunningStatus::IDLE)
	{
		m_chronoPauseLength = 0;
		m_chronoSimStart = std::chrono::steady_clock::now();
	}
	else if (m_nCurrentStatus == ERunningStatus::PAUSED)
		m_chronoPauseLength += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_chronoPauseStart).count();

	// start simulation
	m_nCurrentStatus = ERunningStatus::RUNNING;
	PerformSimulation();

	// performance measurement
	if (m_nCurrentStatus == ERunningStatus::TO_BE_PAUSED)
		m_chronoPauseStart = std::chrono::steady_clock::now();
	else if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED)
	{
		// output performance statistic
		const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_chronoSimStart).count();
		*p_out << "Elapsed time [d:h:m:s]: " << MsToTimeSpan(elapsedTime) << std::endl;
	}

	// set status
	if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED)
		m_nCurrentStatus = ERunningStatus::IDLE;
	else if (m_nCurrentStatus == ERunningStatus::TO_BE_PAUSED)
		m_nCurrentStatus = ERunningStatus::PAUSED;
}

void CCPUSimulator::PerformSimulation()
{
	m_bWallsVelocityChanged = true;

	while (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED && m_nCurrentStatus != ERunningStatus::TO_BE_PAUSED)
	{
		const size_t newObjects = GenerateNewObjects();

		if (std::fabs(m_dCurrentTime - m_dLastSavingTime) + 0.1 * m_currSimulationStep > m_dSavingStep || newObjects)
		{
			SaveData();
			m_dLastSavingTime = m_dCurrentTime;
			PrintStatus();
			if (AdditionalStopCriterionMet())
			{
				m_nCurrentStatus = ERunningStatus::TO_BE_STOPPED;
				break;
			}
		}
		if (m_bAnalyzeCollisions)
			m_CollCalculator.SaveCollisions();

		if (m_Scene.m_PBC.bEnabled)
		{
			m_Scene.m_PBC.UpdatePBC(m_dCurrentTime);
			if (!m_Scene.m_PBC.vVel.IsZero()) // if velocity is non zero
				InitializeModels(); // to set
		}
		LeapFrogStep(m_bPredictionStep);
		if (m_bPredictionStep) // makes prediction step
			m_bPredictionStep = false;
		else
			m_dCurrentTime += m_currSimulationStep;

		// analyze about possible end of interval
		if (m_dCurrentTime >= m_dEndTime)
			m_nCurrentStatus = ERunningStatus::TO_BE_STOPPED;
	}

	if (m_nCurrentStatus == ERunningStatus::TO_BE_STOPPED)	// save current state
	{
		SaveData();
		m_pSystemStructure->SaveToFile();
		if (m_bAnalyzeCollisions)
			m_CollCalculator.SaveRestCollisions();
	}
}

void CCPUSimulator::LeapFrogStep(bool _bPredictionStep /*= false*/)
{
	InitializeStep(m_currSimulationStep);
	CalculateForces(m_currSimulationStep);
	MoveObjects(m_currSimulationStep, _bPredictionStep);
}

void CCPUSimulator::InitializeStep(double _dTimeStep)
{
	m_Scene.ClearAllForcesAndMoments();

	// if there is no contact model, then there is no necessity to calculate contacts
	if (m_pPPModel || m_pPWModel)
	{
		UpdateVerletLists(_dTimeStep); // between PP and PW
		m_CollCalculator.UpdateCollisionMatrixes(_dTimeStep, m_dCurrentTime);
	}
}

void CCPUSimulator::CalculateForces(double _dTimeStep)
{
	if (m_pEFModel) CalculateForcesEF(_dTimeStep);
	if (m_pPPModel) CalculateForcesPP(_dTimeStep);
	if (m_pPWModel) CalculateForcesPW(_dTimeStep);
	if (m_pSBModel) CalculateForcesSB(_dTimeStep);
	if (m_pLBModel) CalculateForcesLB(_dTimeStep);
	m_CollCalculator.CalculateTotalStatisticsInfo();
}

void CCPUSimulator::MoveObjects(double _dTimeStep, bool _bPredictionStep /*= false*/)
{
	if (m_Scene.GetMultiSpheresNumber() == 0)
		MoveParticles(_bPredictionStep);
	else
		MoveMultispheres(_dTimeStep, _bPredictionStep);

	MoveWalls(m_currSimulationStep);
}

void CCPUSimulator::CalculateForcesPP(double _dTimeStep)
{
	m_pPPModel->Precalculate(m_dCurrentTime, _dTimeStep);

	SParticleStruct& particles = m_Scene.GetRefToParticles();
	for (auto& vCollisions : m_vTempCollPPArray)
		for (auto& coll : vCollisions)
			coll.clear();

	ParallelFor(m_CollCalculator.m_vCollMatrixPP.size(), [&](size_t i)
	{
		const size_t nIndex = i%m_nThreadsNumber;
		for (auto& pColl : m_CollCalculator.m_vCollMatrixPP[i])
		{
			m_pPPModel->Calculate(m_dCurrentTime, _dTimeStep, pColl);
			particles.Force(i) += pColl->vTotalForce;
			particles.Moment(i) += pColl->vResultMoment1;

			m_vTempCollPPArray[nIndex][pColl->nDstID % m_nThreadsNumber].push_back(pColl);
		}
	});

	ParallelFor([&](size_t i)
	{
		for (size_t j = 0; j < m_nThreadsNumber; ++j)
			for (auto pColl : m_vTempCollPPArray[j][i])
			{
				particles.Force(pColl->nDstID) -= pColl->vTotalForce;
				particles.Moment(pColl->nDstID) += pColl->vResultMoment2;
			}
	});
}

void CCPUSimulator::CalculateForcesPW(double _dTimeStep)
{
	m_pPWModel->Precalculate(m_dCurrentTime, _dTimeStep);

	SParticleStruct& particles = m_Scene.GetRefToParticles();
	SWallStruct& walls = m_Scene.GetRefToWalls();

	for (auto& vCollisions : m_vTempCollPWArray)
		for (auto& coll : vCollisions)
			coll.clear();

	ParallelFor(m_CollCalculator.m_vCollMatrixPW.size(), [&](size_t i)
	{
		const size_t nIndex = i%m_nThreadsNumber;
		for (auto& pColl : m_CollCalculator.m_vCollMatrixPW[i])
		{
			m_pPWModel->Calculate(m_dCurrentTime, _dTimeStep, pColl);
			particles.Force(i) += pColl->vTotalForce;
			particles.Moment(i) += pColl->vResultMoment1;
			m_vTempCollPWArray[nIndex][pColl->nSrcID % m_nThreadsNumber].push_back(pColl);
		}
	});

	ParallelFor([&](size_t i)
	{
		for (size_t j = 0; j < m_nThreadsNumber; ++j)
			for (auto& pColl : m_vTempCollPWArray[j][i])
				walls.Force(pColl->nSrcID) -= pColl->vTotalForce;
	});
}

void CCPUSimulator::CalculateForcesSB(double _dTimeStep)
{
	if (m_Scene.GetBondsNumber() == 0) return;

	m_pSBModel->Precalculate(m_dCurrentTime, _dTimeStep);

	SParticleStruct& particles = m_Scene.GetRefToParticles();
	SSolidBondStruct& bonds = m_Scene.GetRefToSolidBonds();
	std::vector<std::vector<unsigned>>& partToSolidBonds = *m_Scene.GetPointerToPartToSolidBonds();
	std::vector<unsigned> vBrockenBonds(m_nThreadsNumber, 0);

	ParallelFor(m_Scene.GetBondsNumber(), [&](size_t i)
	{
		if (bonds.Active(i))
			m_pSBModel->Calculate(m_dCurrentTime, _dTimeStep, i, bonds, &vBrockenBonds[i% m_nThreadsNumber]);
	});
	m_nBrockenBondsNumber += VectorSum(vBrockenBonds);

	ParallelFor(partToSolidBonds.size(), [&](size_t i)
	{
		for (size_t j = 0; j < partToSolidBonds[i].size(); j++)
		{
			unsigned nBond = partToSolidBonds[i][j];
			if (!bonds.Active(nBond)) continue;
			if (bonds.LeftID(nBond) == i)
			{
				particles.Force(i) += bonds.TotalForce(nBond);
				particles.Moment(i) += bonds.NormalMoment(nBond) + bonds.TangentialMoment(nBond) - bonds.UnsymMoment(nBond);
			}
			else if (bonds.RightID(nBond) == i)
			{
				particles.Force(i) -= bonds.TotalForce(nBond);
				particles.Moment(i) -= bonds.NormalMoment(nBond) + bonds.TangentialMoment(nBond) + bonds.UnsymMoment(nBond);
			}
		}
	});
}

void CCPUSimulator::CalculateForcesLB(double _dTimeStep)
{
	if (m_Scene.GetLiquidBondsNumber() == 0) return;

	m_pLBModel->Precalculate(m_dCurrentTime, _dTimeStep);

	SLiquidBondStruct& bonds = m_Scene.GetRefToLiquidBonds();
	SParticleStruct& particles = m_Scene.GetRefToParticles();
	std::vector<unsigned> vBrockenBonds(m_nThreadsNumber, 0);

	ParallelFor(m_Scene.GetLiquidBondsNumber(), [&](size_t i)
	{
		if (bonds.Active(i))
			m_pLBModel->Calculate(m_dCurrentTime, _dTimeStep,i, bonds, &vBrockenBonds[i%m_nThreadsNumber]);
	});
	m_nBrockenLiquidBondsNumber += VectorSum(vBrockenBonds);

	for (size_t i = 0; i < m_Scene.GetLiquidBondsNumber(); ++i)
		if (bonds.Active(i))
		{
			particles.Force(bonds.LeftID(i)) += bonds.NormalForce(i) + bonds.TangentialForce(i);
			particles.Force(bonds.RightID(i)) -= bonds.NormalForce(i) + bonds.TangentialForce(i);
			particles.Moment(bonds.LeftID(i)) -= bonds.UnsymMoment(i);
			particles.Moment(bonds.RightID(i)) -= bonds.UnsymMoment(i);
		}
}

void CCPUSimulator::CalculateForcesEF(double _dTimeStep)
{
	m_pEFModel->Precalculate(m_dCurrentTime, _dTimeStep);

	SParticleStruct& particles = m_Scene.GetRefToParticles();

	ParallelFor(particles.Size(), [&](size_t i)
	{
		if (particles.Active(i))
			m_pEFModel->Calculate(m_dCurrentTime, _dTimeStep,i, particles);
	});
}

void CCPUSimulator::MoveParticles(bool _bPredictionStep)
{
	SParticleStruct& particles = m_Scene.GetRefToParticles();

	// apply external acceleration
	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		particles.Force(i) += m_vecExternalAccel * particles.Mass(i);
	});

	// change current simulation time step
	if (m_bVariableTimeStep)
	{
		double maxStep = std::numeric_limits<double>::max();
		for (size_t i = 0; i < particles.Size(); i++)
		{
			if (!particles.Force(i).IsZero())
				maxStep = std::min(maxStep, std::pow(particles.Mass(i), 2.) / particles.Force(i).SquaredLength());
		}
		maxStep = std::sqrt(std::sqrt(maxStep) * m_partMoveLimit);
		if (m_currSimulationStep > maxStep)
			m_currSimulationStep = maxStep;
		else if (m_currSimulationStep < m_initSimulationStep)
			m_currSimulationStep = std::min(m_currSimulationStep * m_timeStepFactor, m_initSimulationStep);
	}
	const double dTimeStep = !_bPredictionStep ? m_currSimulationStep : m_currSimulationStep / 2.;

	// move particles
	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		particles.Vel(i) += particles.Force(i) / particles.Mass(i) * dTimeStep;

		if (m_bConsiderAnisotropy)
		{
			const CMatrix3 rotMatrix = particles.Quaternion(i).ToRotmat();
			CVector3 vTemp = (rotMatrix.Transpose()*particles.Moment(i));
			vTemp.x = vTemp.x / particles.InertiaMoment(i);
			vTemp.y = vTemp.y / particles.InertiaMoment(i);
			vTemp.z = vTemp.z / particles.InertiaMoment(i);
			particles.AnglVel(i) += rotMatrix * vTemp * dTimeStep;
			if (!_bPredictionStep)
			{
				const CVector3& angVel = particles.AnglVel(i);
				CQuaternion& quart = particles.Quaternion(i);
				CQuaternion quaternTemp;
				quaternTemp.q0 = 0.5*dTimeStep*(-quart.q1*angVel.x - quart.q2*angVel.y - quart.q3*angVel.z);
				quaternTemp.q1 = 0.5*dTimeStep*(quart.q0*angVel.x + quart.q3*angVel.y - quart.q2*angVel.z);
				quaternTemp.q2 = 0.5*dTimeStep*(-quart.q3*angVel.x + quart.q0*angVel.y + quart.q1*angVel.z);
				quaternTemp.q3 = 0.5*dTimeStep*(quart.q2*angVel.x - quart.q1*angVel.y + quart.q0*angVel.z);
				quart += quaternTemp;
				quart.Normalize();
			}
		}
		else
			particles.AnglVel(i) += particles.Moment(i) / particles.InertiaMoment(i) * dTimeStep;
		if (!_bPredictionStep)
			particles.Coord(i) += particles.Vel(i)*dTimeStep;
	});

	MoveParticlesOverPBC(); // move virtual particles and check boundaries
}

void CCPUSimulator::MoveWalls(double _dTimeStep)
{
	SWallStruct& pWalls = m_Scene.GetRefToWalls();
	m_bWallsVelocityChanged = false;
	for (size_t i = 0; i < m_pSystemStructure->GetGeometriesNumber(); i++)
	{
		SGeometryObject* pGeom = m_pSystemStructure->GetGeometry(i);
		if (pGeom->vPlanes.empty()) continue;

		if (pGeom->bForceDepVel) // force
		{
			double dTotalForce = 0;
			for (size_t j = 0; j < pGeom->vPlanes.size(); j++)
			{
				size_t nIndex = m_Scene.m_vNewIndexes[pGeom->vPlanes[j]];
				dTotalForce += (pWalls.Force(nIndex).z);
			}
			pGeom->UpdateCurrentInterval(dTotalForce);
		}
		else
			pGeom->UpdateCurrentInterval(m_dCurrentTime); // time
		CVector3 vVel = pGeom->GetCurrentVel();
		CVector3 vRotVel = pGeom->GetCurrentRotVel();
		CVector3 vRotCenter;
		if (pGeom->bRotateAroundCenter)
		{
			vRotCenter.Init(0);
			for (size_t j = 0; j < pGeom->vPlanes.size(); j++)
			{
				size_t nIndex = m_Scene.m_vNewIndexes[pGeom->vPlanes[j]];
				vRotCenter += (pWalls.Vert1(nIndex) + pWalls.Vert2(nIndex) + pWalls.Vert3(nIndex)) / (3.0*pGeom->vPlanes.size());
			}
		}
		else
			vRotCenter = pGeom->GetCurrentRotCenter();

		if (m_dCurrentTime == 0)
			m_bWallsVelocityChanged = true;
		else
		{
			if (!(pGeom->GetCurrentVel() - pGeom->GetCurrentVel()).IsZero())
				m_bWallsVelocityChanged = true;
			else if (!(pGeom->GetCurrentRotVel() - pGeom->GetCurrentRotVel()).IsZero())
				m_bWallsVelocityChanged = true;
			else if (!(pGeom->GetCurrentRotCenter() - pGeom->GetCurrentRotCenter()).IsZero())
				m_bWallsVelocityChanged = true;
		}

		if (!pGeom->vFreeMotion.IsZero() && pGeom->dMass)// solve newtons motion for wall
		{
			CVector3 vTotalForce = m_vecExternalAccel * pGeom->dMass;
			CVector3 vTotalAverVel(0);
			// calculate total force acting on wall
			for (unsigned j = 0; j < pGeom->vPlanes.size(); j++)
			{
				vTotalForce += pWalls.Force(m_Scene.m_vNewIndexes[pGeom->vPlanes[j]]);
				vTotalAverVel += pWalls.Vel(m_Scene.m_vNewIndexes[pGeom->vPlanes[j]]) / static_cast<double>(pGeom->vPlanes.size());
			}
			if (pGeom->vFreeMotion.x)
				vVel.x = vTotalAverVel.x + _dTimeStep * vTotalForce.x / pGeom->dMass;
			if (pGeom->vFreeMotion.y)
				vVel.y = vTotalAverVel.y + _dTimeStep * vTotalForce.y / pGeom->dMass;
			if (pGeom->vFreeMotion.z)
				vVel.z = vTotalAverVel.z + _dTimeStep * vTotalForce.z / pGeom->dMass;
			m_bWallsVelocityChanged = true;
		}

		if (vVel.IsZero() && vRotVel.IsZero()) continue;
		CMatrix3 RotMatrix;
		if (!vRotVel.IsZero())
			RotMatrix = CQuaternion(vRotVel*_dTimeStep).ToRotmat();

		ParallelFor(pGeom->vPlanes.size(), [&](size_t j)
		{
			size_t nIndex = m_Scene.m_vNewIndexes[pGeom->vPlanes[j]];
			pWalls.Vel(nIndex) = vVel;
			pWalls.RotVel(nIndex) = vRotVel;
			pWalls.RotCenter(nIndex) = vRotCenter;
			if (!vVel.IsZero())
			{
				pWalls.Vert1(nIndex) += vVel * _dTimeStep;
				pWalls.Vert2(nIndex) += vVel * _dTimeStep;
				pWalls.Vert3(nIndex) += vVel * _dTimeStep;
			}

			if (!vRotVel.IsZero())
			{
				pWalls.Vert1(nIndex) = vRotCenter + RotMatrix * (pWalls.Vert1(nIndex) - vRotCenter);
				pWalls.Vert2(nIndex) = vRotCenter + RotMatrix * (pWalls.Vert2(nIndex) - vRotCenter);
				pWalls.Vert3(nIndex) = vRotCenter + RotMatrix * (pWalls.Vert3(nIndex) - vRotCenter);
			}
			// update wall properties
			pWalls.MinCoord(nIndex) = Min(pWalls.Vert1(nIndex), pWalls.Vert2(nIndex), pWalls.Vert3(nIndex));
			pWalls.MaxCoord(nIndex) = Max(pWalls.Vert1(nIndex), pWalls.Vert2(nIndex), pWalls.Vert3(nIndex));

			if (!vRotVel.IsZero())
				pWalls.NormalVector(nIndex) = Normalized((pWalls.Vert2(nIndex) - pWalls.Vert1(nIndex))*(pWalls.Vert3(nIndex) - pWalls.Vert1(nIndex)));
		});
	}
}

void CCPUSimulator::MoveMultispheres(double _dTimeStep, bool _bPredictionStep)
{
	SMultiSphere& pMultispheres = m_Scene.GetRefToMultispheres();
	SParticleStruct& pParticles = m_Scene.GetRefToParticles();

	// move particles which does not correlated to any multisphere
	ParallelFor(pMultispheres.Size(), [&](size_t i)
	{
		if (pParticles.MultiSphIndex(i) == -1)
		{
			pParticles.Force(i) += m_vecExternalAccel * pParticles.Mass(i);
			pParticles.Vel(i) += pParticles.Force(i) / pParticles.Mass(i) * _dTimeStep;
			pParticles.AnglVel(i) += pParticles.Moment(i) / pParticles.InertiaMoment(i) * _dTimeStep;
			if (!_bPredictionStep)
				pParticles.Coord(i) += pParticles.Vel(i)*_dTimeStep;
		}
	});

	// move all multispheres
	ParallelFor(pMultispheres.Size(), [&](size_t i)
	{
		CVector3 vTotalForce(0), vTotalMoment(0);

		for (unsigned j = 0; j < pMultispheres.Indices(i).size(); j++)
		{
			vTotalForce += pParticles.Force(pMultispheres.Indices(i)[j]);
			vTotalMoment += (pParticles.Coord(pMultispheres.Indices(i)[j]) - pMultispheres.Center(i))*pParticles.Force(pMultispheres.Indices(i)[j]);
		}
		vTotalForce += m_vecExternalAccel * pMultispheres.Mass(i);

		pMultispheres.Velocity(i) += vTotalForce / pMultispheres.Mass(i)*_dTimeStep;
		CVector3 vAngle = pMultispheres.InvLMatrix(i)*pMultispheres.RotVelocity(i)*_dTimeStep;
		double dAngleLSquared = vAngle.SquaredLength();
		CMatrix3 newLMatrix, newInvLMatrix, deltaL;
		if (dAngleLSquared > 0)
		{
			double dAngleL = sqrt(dAngleLSquared);
			double dCosPhi = cos(dAngleL);
			double dSinPhi = sin(dAngleL);
			deltaL.values[0][0] = vAngle.x*vAngle.x / dAngleLSquared + dCosPhi * (1 - vAngle.x*vAngle.x / dAngleLSquared);
			deltaL.values[0][1] = vAngle.x*vAngle.y / dAngleLSquared * (1 - dCosPhi) - vAngle.z*dSinPhi / dAngleL;
			deltaL.values[0][2] = vAngle.x*vAngle.z / dAngleLSquared * (1 - dCosPhi) + vAngle.y*dSinPhi / dAngleL;

			deltaL.values[1][0] = vAngle.y*vAngle.x / dAngleLSquared * (1 - dCosPhi) + vAngle.z*dSinPhi / dAngleL;
			deltaL.values[1][1] = vAngle.y*vAngle.y / dAngleLSquared + dCosPhi * (1 - vAngle.y*vAngle.y / dAngleLSquared);
			deltaL.values[1][2] = vAngle.y*vAngle.z / dAngleLSquared * (1 - dCosPhi) - vAngle.x*dSinPhi / dAngleL;

			deltaL.values[2][0] = vAngle.z*vAngle.x / dAngleLSquared * (1 - dCosPhi) - vAngle.y*dSinPhi / dAngleL;
			deltaL.values[2][1] = vAngle.z*vAngle.y / dAngleLSquared * (1 - dCosPhi) + vAngle.x*dSinPhi / dAngleL;
			deltaL.values[2][2] = vAngle.z*vAngle.z / dAngleLSquared + dCosPhi * (1 - vAngle.z*vAngle.z / dAngleLSquared);

			newLMatrix = deltaL * pMultispheres.LMatrix(i);
			newInvLMatrix = newLMatrix.Inverse();
			pMultispheres.RotVelocity(i) = (newLMatrix*pMultispheres.InvInertTensor(i)*newInvLMatrix)*
				(pMultispheres.LMatrix(i)*pMultispheres.InertTensor(i)*pMultispheres.InvLMatrix(i)*pMultispheres.RotVelocity(i) + vTotalMoment * _dTimeStep);

			pMultispheres.LMatrix(i) = newLMatrix;
			pMultispheres.InvLMatrix(i) = newInvLMatrix;
		}
		else
			pMultispheres.RotVelocity(i) = (pMultispheres.LMatrix(i)*pMultispheres.InvInertTensor(i)*pMultispheres.InvLMatrix(i))*(vTotalMoment*_dTimeStep);

		if (!_bPredictionStep)
		{
			CVector3 vOldCoord = pMultispheres.Center(i);
			pMultispheres.Center(i) += pMultispheres.Velocity(i)*_dTimeStep;
			for (unsigned j = 0; j < pMultispheres.Indices(i).size(); j++)
			{
				if (dAngleLSquared > 0)
					pParticles.Coord(pMultispheres.Indices(i)[j]) = pMultispheres.Center(i) +
					deltaL * (pParticles.Coord(pMultispheres.Indices(i)[j]) - vOldCoord);
				else
					pParticles.Coord(pMultispheres.Indices(i)[j]) = pMultispheres.Center(i) + pParticles.Coord(pMultispheres.Indices(i)[j]) - vOldCoord;

				pParticles.Vel(pMultispheres.Indices(i)[j]) = pMultispheres.Velocity(i) +
					pMultispheres.RotVelocity(i)*(pParticles.Coord(pMultispheres.Indices(i)[j]) - pMultispheres.Center(i));
				pParticles.AnglVel(pMultispheres.Indices(i)[j]) = pMultispheres.RotVelocity(i);
			}
		}
	});
}

void CCPUSimulator::InitializeModels()
{
	p_InitializeModels();

	for (auto& model : m_models)
		model->Initialize(
			m_Scene.GetPointerToParticles().get(),
			m_Scene.GetPointerToWalls().get(),
			m_Scene.GetPointerToSolidBonds().get(),
			m_Scene.GetPointerToLiquidBonds().get(),
			m_Scene.GetPointerToInteractProperties().get());
}

size_t CCPUSimulator::GenerateNewObjects()
{
	return p_GenerateNewObjects();
}

void CCPUSimulator::PrepareAdditionalSavingData()
{
	SParticleStruct& particles = m_Scene.GetRefToParticles();

	// reset previously calculated stresses
	for (size_t i = 0; i < m_vAddSavingDataPart.size(); ++i)
		m_vAddSavingDataPart[i].sStressTensor.Init(0);

	// save stresses caused by solid bonds
	SSolidBondStruct& solidBonds = m_Scene.GetRefToSolidBonds();
	for (size_t i = 0; i < solidBonds.Size(); ++i)
	{
		CSolidBond* pSBond = static_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex(solidBonds.InitIndex(i)));
		if ((!solidBonds.Active(i)) && (!pSBond->IsActive(m_dCurrentTime)))
			continue;
		const size_t leftID = solidBonds.LeftID(i);
		const size_t rightID = solidBonds.RightID(i);
		CVector3 vConnVec = (particles.Coord(leftID) - particles.Coord(rightID)).Normalized();
		m_vAddSavingDataPart[leftID].AddStress(-1 * vConnVec * particles.Radius(leftID), solidBonds.TotalForce(i), PI * pow(2 * particles.Radius(leftID), 3) / 6);
		m_vAddSavingDataPart[rightID].AddStress(vConnVec * particles.Radius(rightID), -1 * solidBonds.TotalForce(i), PI * pow(2 * particles.Radius(rightID), 3) / 6);
	}

	// save stresses caused by particle-particle contact
	for (size_t i = 0; i < m_CollCalculator.m_vCollMatrixPP.size(); i++)
	{
		for (auto& pColl : m_CollCalculator.m_vCollMatrixPP[i])
		{
			const size_t srcID = pColl->nSrcID;
			const size_t dstID = pColl->nDstID;
			CVector3 vConnVec = (particles.Coord(srcID) - particles.Coord(dstID)).Normalized();
			const double srcRadius = particles.Radius(srcID);
			const double dstRadius = particles.Radius(dstID);
			m_vAddSavingDataPart[pColl->nSrcID].AddStress(-1 * vConnVec*srcRadius, pColl->vTotalForce, PI * pow(2 * srcRadius, 3) / 6);
			m_vAddSavingDataPart[pColl->nDstID].AddStress(vConnVec*dstRadius, -1 * pColl->vTotalForce, PI * pow(2 * dstRadius, 3) / 6);
		}
	};

	// save stresses caused by particle-wall contacts
	for (size_t i = 0; i < m_CollCalculator.m_vCollMatrixPW.size(); i++)
	{
		for (auto& pColl : m_CollCalculator.m_vCollMatrixPW[i])
		{
			CVector3 vConnVec = (pColl->vContactVector - particles.Coord(pColl->nDstID)).Normalized();
			m_vAddSavingDataPart[pColl->nDstID].AddStress(vConnVec*particles.Radius(pColl->nDstID), pColl->vTotalForce, PI * pow(2 * particles.Radius(pColl->nDstID), 3) / 6);
		}
	};
}

void CCPUSimulator::SaveData()
{
	const clock_t t = clock();
	p_SaveData();
	m_VerletList.AddDisregardingTimeInterval(clock() - t);
}

void CCPUSimulator::UpdateVerletLists(double _dTimeStep)
{
	// update max velocity
	m_dMaxParticleVelocity = m_Scene.GetMaxParticleVelocity();
	if (m_bWallsVelocityChanged)
		m_dMaxWallVelocity = m_Scene.GetMaxWallVelocity();
	CheckParticlesInDomain();
	if (m_VerletList.IsNeedToBeUpdated(_dTimeStep, m_Scene.GetMaxPartVerletDistance(), m_dMaxWallVelocity))
	{
		m_VerletList.UpdateList(m_dCurrentTime);
		m_Scene.SaveVerletCoords();
	}
}

void CCPUSimulator::CheckParticlesInDomain()
{
	SVolumeType simDomain = m_pSystemStructure->GetSimulationDomain();
	SParticleStruct& particles = m_Scene.GetRefToParticles();
	SSolidBondStruct& solidBonds = m_Scene.GetRefToSolidBonds();
	SLiquidBondStruct& liquidBonds = m_Scene.GetRefToLiquidBonds();
	std::vector<size_t> vInactiveParticlesNum(m_Scene.GetTotalParticlesNumber());
	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		if ((particles.Active(i)) && (!IsPointInDomain(simDomain, particles.Coord(i)))) // remove particles situated not in the domain
		{
			particles.Active(i) = false;
			particles.EndActivity(i) = m_dCurrentTime;
			vInactiveParticlesNum[i]++;
			for (size_t j = 0; j < solidBonds.Size(); ++j) // delete all bonds that connected to this particle
				if ((solidBonds.Active(j)) && ((solidBonds.LeftID(j) == i) || (solidBonds.RightID(j) == i)))
				{
					solidBonds.Active(j) = false;
					solidBonds.EndActivity(j) = m_dCurrentTime;
				}
			for (size_t j = 0; j < liquidBonds.Size(); ++j) // delete all bonds that connected to this particle
				if ((liquidBonds.Active(j)) && ((liquidBonds.LeftID(j) == i) || (liquidBonds.RightID(j) == i)))
				{
					liquidBonds.Active(j) = false;
					liquidBonds.EndActivity(j) = m_dCurrentTime;
				}
		}
	});

	const size_t nNewInactiveParticles = VectorSum(vInactiveParticlesNum);
	m_nInactiveParticlesNumber += nNewInactiveParticles;
	if (nNewInactiveParticles > 0)
		m_Scene.UpdateParticlesToBonds();
}

void CCPUSimulator::MoveParticlesOverPBC()
{
	const SPBC& pbc = m_Scene.m_PBC;
	if (!pbc.bEnabled) return;
	SParticleStruct& particles = m_Scene.GetRefToParticles();

	//use <unsigned char> in order to avoid problems with simultaneous writing from several threads
	//std::vector<unsigned char> vBoundaryCrossed(m_Scene.GetTotalParticlesNumber());	// whether the particle crossed the PBC boundary
	std::vector<uint8_t> vShifts(m_Scene.GetTotalParticlesNumber(), 0);	// whether the particle crossed the PBC boundary

	// TODO: It is unnecessary to analyze all particles for all crosses. Only those should be analyzed, who can cross boundary during current verlet step.
	// shift particles if they crossed boundary
	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		CVector3& vCoord = particles.Coord(i);
		// particle crossed left boundary
		if (pbc.bX && vCoord.x <= pbc.currentDomain.coordBeg.x)  vShifts[i] = vShifts[i] | 32;
		if (pbc.bY && vCoord.y <= pbc.currentDomain.coordBeg.y) vShifts[i] = vShifts[i] | 8;
		if (pbc.bZ && vCoord.z <= pbc.currentDomain.coordBeg.z) vShifts[i] = vShifts[i] | 2;
		// particle crossed right boundary
		if (pbc.bX && vCoord.x >= pbc.currentDomain.coordEnd.x) vShifts[i] = vShifts[i] | 16;
		if (pbc.bY && vCoord.y >= pbc.currentDomain.coordEnd.y) vShifts[i] = vShifts[i] | 4;
		if (pbc.bZ && vCoord.z >= pbc.currentDomain.coordEnd.z) vShifts[i] = vShifts[i] | 1;

		if (vShifts[i])
		{
			vCoord += GetVectorFromVirtShift(vShifts[i], m_Scene.m_PBC.boundaryShift);
			particles.CoordVerlet(i) += GetVectorFromVirtShift(vShifts[i], m_Scene.m_PBC.boundaryShift);
		}
	});

	// this can be in case when all contact models are turned off
	if (m_VerletList.m_PPList.empty() && m_VerletList.m_PWList.empty()) return;

	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		// modify shift in possible particle-particle contacts
		for (size_t j = 0; j < m_VerletList.m_PPList[i].size(); j++)
		{
			const size_t srcID = i;
			const size_t dstID = m_VerletList.m_PPList[i][j];
			if (vShifts[srcID])
				m_VerletList.m_PPVirtShift[i][j] = AddVirtShift(m_VerletList.m_PPVirtShift[i][j], vShifts[srcID]);
			if (vShifts[dstID])
				m_VerletList.m_PPVirtShift[i][j] = SubstractVirtShift(m_VerletList.m_PPVirtShift[i][j], vShifts[dstID]);
		}

		// modify shift in existing particle-particle  collisions
		for (size_t j = 0; j < m_CollCalculator.m_vCollMatrixPP[i].size(); j++)
		{
			const unsigned srcID = m_CollCalculator.m_vCollMatrixPP[i][j]->nSrcID;
			const unsigned dstID = m_CollCalculator.m_vCollMatrixPP[i][j]->nDstID;
			if (vShifts[srcID])
				m_CollCalculator.m_vCollMatrixPP[i][j]->nVirtShift = AddVirtShift(m_CollCalculator.m_vCollMatrixPP[i][j]->nVirtShift, vShifts[srcID]);
			if (vShifts[dstID])
				m_CollCalculator.m_vCollMatrixPP[i][j]->nVirtShift = SubstractVirtShift(m_CollCalculator.m_vCollMatrixPP[i][j]->nVirtShift, vShifts[dstID]);
		}

		// modify shift in possible particle-wall contacts
		for (size_t j = 0; j < m_VerletList.m_PWList[i].size(); j++)
			if (vShifts[i])
				m_VerletList.m_PWVirtShift[i][j] = SubstractVirtShift(m_VerletList.m_PWVirtShift[i][j], vShifts[i]);

		// modify shift in existing particle-wall collisions
		for (size_t j = 0; j < m_CollCalculator.m_vCollMatrixPW[i].size(); j++)
			if (vShifts[i])
				m_CollCalculator.m_vCollMatrixPW[i][j]->nVirtShift = SubstractVirtShift(m_CollCalculator.m_vCollMatrixPW[i][j]->nVirtShift, vShifts[i]);
	});
}
