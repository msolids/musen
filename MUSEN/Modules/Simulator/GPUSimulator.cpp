/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GPUSimulator.h"

CGPUSimulator::CGPUSimulator()
{
	m_cudaDefines = new CCUDADefines;
	m_cudaDefines->Initialize();
	m_gpu.SetCudaDefines(m_cudaDefines);
	m_SceneGPU.SetCudaDefines(m_cudaDefines);
	InitializeSimulator();
}

CGPUSimulator::CGPUSimulator(const CBaseSimulator& _simulator)
{
	m_cudaDefines = new CCUDADefines;
	m_cudaDefines->Initialize();
	m_gpu.SetCudaDefines(m_cudaDefines);
	m_SceneGPU.SetCudaDefines(m_cudaDefines);
	p_CopySimulatorData(_simulator);
	InitializeSimulator();
}

CGPUSimulator::~CGPUSimulator()
{
	if (m_pInteractProps)
		CUDA_FREE_D(m_pInteractProps);

	CUDA_FREE_D(m_pDispatchedResults_d);
	CUDA_FREE_H(m_pDispatchedResults_h);
}

void CGPUSimulator::InitializeSimulator()
{
	m_gpu.SetExternalAccel(m_vecExternalAccel);
	m_gpu.SetPBC(m_Scene.m_PBC);
	m_gpu.SetAnisotropyFlag(m_bConsiderAnisotropy);

	CUDA_MALLOC_D(&m_pDispatchedResults_d, sizeof(SDispatchedResults));
	CUDA_MALLOC_H(&m_pDispatchedResults_h, sizeof(SDispatchedResults));

	m_pInteractProps = nullptr;
}

void CGPUSimulator::SetExternalAccel(const CVector3& _accel)
{
	CBaseSimulator::SetExternalAccel(_accel);
	m_gpu.SetExternalAccel(m_vecExternalAccel);
}

void CGPUSimulator::GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID)
{
	m_gpu.GetOverlapsInfo(m_SceneGPU.GetPointerToParticles(), _nMaxParticleID, _dMaxOverlap, _dAverageOverlap);
}

void CGPUSimulator::StartSimulation()
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

		m_gpu.SetSimulationDomain(m_pSystemStructure->GetSimulationDomain());

		// get flag of anisotropy
		m_bConsiderAnisotropy = m_pSystemStructure->IsAnisotropyEnabled();
		m_gpu.SetAnisotropyFlag(m_bConsiderAnisotropy);

		// delete old data
		m_pSystemStructure->ClearAllStatesFrom(m_dCurrentTime);

		// Initialize scene on CPU
		m_Scene.SetSystemStructure(m_pSystemStructure);
		m_Scene.InitializeScene(m_dCurrentTime,GetModelManager()->GetUtilizedVariables());

		// Initialize scene, PBC, models on GPU
		m_SceneGPU.CUDASaveVerletCoords();
		m_SceneGPU.InitializeScene(m_Scene, m_pSystemStructure);
		m_gpu.SetPBC(m_Scene.m_PBC);
		InitializeModels();

		m_pGenerationManager->Initialize();

		// initialize verlet list
		m_VerletList.InitializeList();
		m_VerletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_Scene.GetMinParticleContactRadius(), m_Scene.GetMaxParticleContactRadius(),
								m_nCellsMax, m_dVerletDistanceCoeff, m_bAutoAdjustVerletDistance);
		m_VerletList.ResetCurrentData();

		CUDAInitializeWalls();
		m_gpu.InitializeCollisions();
		CUDAInitializeMaterials();
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

void CGPUSimulator::PerformSimulation()
{
	m_bWallsVelocityChanged = true;

	while (m_nCurrentStatus != ERunningStatus::TO_BE_STOPPED && m_nCurrentStatus != ERunningStatus::TO_BE_PAUSED)
	{
		GenerateNewObjects();

		if (fabs(m_dCurrentTime - m_dLastSavingTime) + 0.1 * m_currSimulationStep > m_dSavingStep)
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

		if (m_Scene.m_PBC.bEnabled)
		{
			m_Scene.m_PBC.UpdatePBC(m_dCurrentTime);
			if (!m_Scene.m_PBC.vVel.IsZero()) // if velocity is non zero
			{
				InitializeModels();
				m_gpu.SetPBC(m_Scene.m_PBC);
			}
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
	}
}

void CGPUSimulator::LeapFrogStep(bool _bPredictionStep)
{
	InitializeStep(m_currSimulationStep);
	CalculateForces(m_currSimulationStep);
	MoveObjects(m_currSimulationStep, _bPredictionStep);
}

void CGPUSimulator::InitializeStep(double _dTimeStep)
{
	// clear all forces and moment on GPU
	m_SceneGPU.ClearAllForcesAndMoments();

	// if there is no contact model, then there is no necessity to calculate contacts
	if (m_pPPModel || m_pPWModel)
	{
		UpdateVerletLists(_dTimeStep); // between PP and PW
		CUDAUpdateActiveCollisions();
	}
}

void CGPUSimulator::CalculateForces(double _dTimeStep)
{
	if (m_pPPModel) CalculateForcesPP(_dTimeStep);
	if (m_pPWModel) CalculateForcesPW(_dTimeStep);
	if (m_pSBModel) CalculateForcesSB(_dTimeStep);
	if (m_pLBModel) CalculateForcesLB(_dTimeStep);
	if (m_pEFModel) CalculateForcesEF(_dTimeStep);
	cudaStreamQuery(0);
}

void CGPUSimulator::MoveObjects(double _dTimeStep, bool _bPredictionStep)
{
	MoveParticles(_bPredictionStep); // Recalculation of time step
	MoveWalls(_dTimeStep);
}

void CGPUSimulator::CalculateForcesPP(double _dTimeStep)
{
	if (!m_gpu.m_CollisionsPP.collisions.nElements) return;
	m_pPPModel->CalculatePPForceGPU(m_dCurrentTime, _dTimeStep, m_pInteractProps, m_SceneGPU.GetPointerToParticles(), m_gpu.m_CollisionsPP.collisions);
}

void CGPUSimulator::CalculateForcesPW(double _dTimeStep)
{
	if (!m_gpu.m_CollisionsPW.collisions.nElements) return;
	m_pPWModel->CalculatePWForceGPU(m_dCurrentTime, _dTimeStep, m_pInteractProps,
		m_SceneGPU.GetPointerToParticles(), m_SceneGPU.GetPointerToWalls(), m_gpu.m_CollisionsPW.collisions);
	m_gpu.GatherForcesFromPWCollisions(m_SceneGPU.GetPointerToParticles(), m_SceneGPU.GetPointerToWalls());
}

void CGPUSimulator::CalculateForcesSB(double _dTimeStep)
{
	if (m_Scene.GetBondsNumber() == 0) return;
	m_pSBModel->CalculateSBForceGPU(m_dCurrentTime, _dTimeStep, m_SceneGPU.GetPointerToParticles(), m_SceneGPU.GetPointerToSolidBonds());
}

void CGPUSimulator::CalculateForcesEF(double _dTimeStep)
{
	m_pEFModel->CalculateEFForceGPU(m_dCurrentTime, _dTimeStep, m_SceneGPU.GetPointerToParticles());
}

void CGPUSimulator::MoveParticles(bool _bPredictionStep)
{
	if (!m_vecExternalAccel.IsZero())
		m_gpu.ApplyExternalAcceleration(m_SceneGPU.GetPointerToParticles());
	if (!_bPredictionStep)
	{
		if (m_bVariableTimeStep)
			m_currSimulationStep = m_gpu.CalculateNewTimeStep(m_currSimulationStep, m_initSimulationStep, m_partMoveLimit, m_timeStepFactor, m_SceneGPU.GetPointerToParticles());
		m_gpu.MoveParticles(m_currSimulationStep, m_initSimulationStep, m_SceneGPU.GetPointerToParticles(), m_bVariableTimeStep);
	}
	else
		m_gpu.MoveParticlesPrediction(m_currSimulationStep / 2., m_SceneGPU.GetPointerToParticles());

	MoveParticlesOverPBC(); // move virtual particles and check boundaries
}

void CGPUSimulator::MoveWalls(double _dTimeStep)
{
	m_bWallsVelocityChanged = false;
	// analysis of transition to new interval
	for (unsigned iGeom = 0; iGeom < m_pSystemStructure->GetGeometriesNumber(); ++iGeom)
	{
		SGeometryObject* pGeom = m_pSystemStructure->GetGeometry(iGeom);

		if (pGeom->vPlanes.empty()) continue;
		if (pGeom->bForceDepVel) // force
		{
			const CVector3 vTotalForce = m_gpu.CalculateTotalForceOnWall(iGeom, m_SceneGPU.GetPointerToWalls());
			pGeom->UpdateCurrentInterval(vTotalForce.z);
		}
		else
			pGeom->UpdateCurrentInterval(m_dCurrentTime); // time

		CVector3 vVel = pGeom->GetCurrentVel();
		CVector3 vRotVel = pGeom->GetCurrentRotVel();
		CVector3 vRotCenter = pGeom->GetCurrentRotCenter();

		if (m_dCurrentTime == 0 || vVel != pGeom->GetCurrentVel() || vRotVel != pGeom->GetCurrentRotVel() || vRotCenter != pGeom->GetCurrentRotCenter())
			m_bWallsVelocityChanged = true;

		if ( !pGeom->vFreeMotion.IsZero() )
			m_bWallsVelocityChanged = true;

		if (vRotVel.IsZero() && pGeom->vFreeMotion.IsZero() && vVel.IsZero()) continue;
		CMatrix3 RotMatrix;
		if (!vRotVel.IsZero())
			RotMatrix = CQuaternion(vRotVel*_dTimeStep).ToRotmat();

		m_gpu.MoveWalls(_dTimeStep, iGeom, vVel, vRotVel, vRotCenter, RotMatrix, pGeom->vFreeMotion,
			pGeom->bForceDepVel, pGeom->bRotateAroundCenter, pGeom->dMass, m_SceneGPU.GetPointerToWalls(), m_vecExternalAccel);
	}
}

void CGPUSimulator::InitializeModels()
{
	p_InitializeModels();
	InitializeModelParameters();
}

void CGPUSimulator::InitializeModelParameters()
{
	for (auto& model : m_models)
		model->InitializeGPU(m_cudaDefines);
}

void CGPUSimulator::GenerateNewObjects()
{
	// TODO: find why contacts are not being detected. Check generation of agglomerates.
	//if (!m_pGenerationManager->IsNeedToBeGenerated(m_dCurrentTime)) return; // no need to generate new objects

	//// get actual data from device
	//m_SceneGPU.CUDAParticlesGPU2CPU(&m_Scene);
	//m_SceneGPU.CUDABondsGPU2CPU(&m_Scene);
	//m_SceneGPU.CUDAWallsGPU2CPU(&m_Scene);

	//if (p_GenerateNewObjects()) // new objects have been generated
	//{
	//	// update data on device
	//	m_SceneGPU.CUDAParticlesCPU2GPU(&m_Scene);
	//	m_SceneGPU.CUDABondsCPU2GPU(&m_Scene);

	//	CUDAInitializeMaterials();
	//}
}

void CGPUSimulator::PrepareAdditionalSavingData()
{
	SParticleStruct& particles = m_Scene.GetRefToParticles();
	SSolidBondStruct& bonds = m_Scene.GetRefToSolidBonds();

	static SGPUCollisions PPCollisions(SBasicGPUStruct::EMemType::HOST), PWCollisions(SBasicGPUStruct::EMemType::HOST);
	m_gpu.CopyCollisionsGPU2CPU(PPCollisions, PWCollisions);

	// reset previously calculated stresses
	for (auto& data : m_vAddSavingDataPart)
		data.sStressTensor.Init(0);

	// save stresses caused by solid bonds
	for (size_t i = 0; i < bonds.Size(); i++)
	{
		CSolidBond* pSBond = dynamic_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex(bonds.InitIndex(i)));
		if (!bonds.Active(i) && !pSBond->IsActive(m_dCurrentTime)) continue;
		const size_t leftID  = bonds.LeftID(i);
		const size_t rightID = bonds.RightID(i);
		CVector3 vConnVec = (particles.Coord(leftID) - particles.Coord(rightID)).Normalized();
		m_vAddSavingDataPart[bonds.LeftID(i)].AddStress(-1 * vConnVec * particles.Radius(leftID), bonds.TotalForce(i), PI * pow(2 * particles.Radius(leftID), 3) / 6);
		m_vAddSavingDataPart[bonds.RightID(i)].AddStress(vConnVec * particles.Radius(rightID), -1 * bonds.TotalForce(i), PI * pow(2 * particles.Radius(rightID), 3) / 6);
	}

	// save stresses caused by particle-particle contact
	for (size_t i = 0; i < PPCollisions.nElements; i++)
	{
		if (!PPCollisions.ActivityFlags[i]) continue;
		const size_t srcID = PPCollisions.SrcIDs[i];
		const size_t dstID = PPCollisions.DstIDs[i];
		CVector3 vConnVec = (particles.Coord(srcID) - particles.Coord(dstID)).Normalized();
		m_vAddSavingDataPart[PPCollisions.SrcIDs[i]].AddStress(-1 * vConnVec*particles.Radius(srcID), PPCollisions.TotalForces[i], PI * pow(2 * particles.Radius(srcID), 3) / 6);
		m_vAddSavingDataPart[PPCollisions.DstIDs[i]].AddStress(vConnVec*particles.Radius(dstID), -1 * PPCollisions.TotalForces[i], PI * pow(2 * particles.Radius(dstID), 3) / 6);
	};

	// save stresses caused by particle-wall contacts
	for (size_t i = 0; i < PWCollisions.nElements; i++)
	{
		if (!PWCollisions.ActivityFlags[i]) continue;
		CVector3 vConnVec = (PWCollisions.ContactVectors[i] - particles.Coord(PWCollisions.DstIDs[i])).Normalized();
		m_vAddSavingDataPart[PWCollisions.DstIDs[i]].AddStress(vConnVec* particles.Radius(PWCollisions.DstIDs[i]), PWCollisions.TotalForces[i], PI * pow(2 * particles.Radius(PWCollisions.DstIDs[i]), 3) / 6);
	};
}

void CGPUSimulator::SaveData()
{
	clock_t t = clock();
	cudaDeviceSynchronize();
	m_SceneGPU.CUDABondsGPU2CPU( m_Scene );
	m_SceneGPU.CUDAParticlesGPU2CPUAllData(m_Scene);
	m_SceneGPU.CUDAWallsGPU2CPUAllData(m_Scene);
	m_nBrockenBondsNumber = m_SceneGPU.GetBrokenBondsNumber();
	m_dMaxParticleVelocity = m_SceneGPU.GetMaxPartVelocity();
	p_SaveData();

	m_VerletList.AddDisregardingTimeInterval(clock() - t);
}

void CGPUSimulator::UpdateVerletLists(double _dTimeStep)
{
	CUDAUpdateGlobalCPUData();
	if (m_VerletList.IsNeedToBeUpdated(_dTimeStep, sqrt(m_pDispatchedResults_h->dMaxSquaredPartDist) , m_dMaxWallVelocity))
	{
		m_SceneGPU.CUDAParticlesGPU2CPUVerletData(m_Scene);
		m_SceneGPU.CUDAWallsGPU2CPUVerletData(m_Scene);
		m_VerletList.UpdateList(m_dCurrentTime);

		static STempStorage storePP, storePW; // to reuse memory
		CUDAUpdateVerletLists(m_VerletList.m_PPList, m_VerletList.m_PPVirtShift, m_gpu.m_CollisionsPP, storePP, true);
		CUDAUpdateVerletLists(m_VerletList.m_PWList, m_VerletList.m_PWVirtShift, m_gpu.m_CollisionsPW, storePW, false);
		m_SceneGPU.CUDASaveVerletCoords();
	}
}

void CGPUSimulator::CUDAUpdateGlobalCPUData()
{
	// check that all particles are remains in simulation domain
	m_gpu.CheckParticlesInDomain(m_dCurrentTime, m_SceneGPU.GetPointerToParticles(), &m_pDispatchedResults_d->nActivePartNum);

	// update max velocities
	m_SceneGPU.GetMaxSquaredPartDist(&m_pDispatchedResults_d->dMaxSquaredPartDist);
	if (m_bWallsVelocityChanged)
		m_SceneGPU.GetMaxWallVelocity(&m_pDispatchedResults_d->dMaxWallVel);

	CUDA_MEMCPY_D2H(m_pDispatchedResults_h, m_pDispatchedResults_d, sizeof(SDispatchedResults));

	const bool bNewInactiveParticles = (m_nInactiveParticlesNumber != m_SceneGPU.GetParticlesNumber() - m_pDispatchedResults_h->nActivePartNum);
	if (bNewInactiveParticles && m_SceneGPU.GetBondsNumber())
	{
		m_gpu.CheckBondsActivity(m_dCurrentTime, m_SceneGPU.GetPointerToParticles(), m_SceneGPU.GetPointerToSolidBonds());
		m_SceneGPU.CUDABondsActivityGPU2CPU(m_Scene);
		m_Scene.UpdateParticlesToBonds();
	}

	m_nInactiveParticlesNumber = m_SceneGPU.GetParticlesNumber() - m_pDispatchedResults_h->nActivePartNum;
	if (m_bWallsVelocityChanged)
		m_dMaxWallVelocity = m_pDispatchedResults_h->dMaxWallVel;
}

void CGPUSimulator::CUDAUpdateActiveCollisions()
{
	m_gpu.UpdateActiveCollisionsPP(m_SceneGPU.GetPointerToParticles());
	m_gpu.UpdateActiveCollisionsPW(m_SceneGPU.GetPointerToParticles(), m_SceneGPU.GetPointerToWalls());
}

void CGPUSimulator::CUDAUpdateVerletLists(const std_matr_u& _verletListCPU, const std_matr_u8& _verletListShiftsCPU,
	CGPU::SCollisionsHolder& _collisions, STempStorage& _store, bool _bPPVerlet)
{
	const size_t nParticles = m_Scene.GetTotalParticlesNumber();
	// calculate total number of possible contacts
	size_t nCollsions = 0;
	for (const auto& colls : _verletListCPU)
		nCollsions += colls.size();

	// create verlet lists for GPU
	_store.hvVerletPartInd.resize(nParticles + 1);
	_store.hvVerletDst.resize(nCollsions);
	_store.hvVerletSrc.resize(nCollsions);
	_store.hvVirtShifts.resize(nCollsions);

	if (!_store.hvVerletPartInd.empty())	_store.hvVerletPartInd.front() = 0;			// for easier access
	for (size_t i = 1; i < nParticles; ++i)
		_store.hvVerletPartInd[i] = _store.hvVerletPartInd[i - 1] + _verletListCPU[i-1].size();
	if (!_store.hvVerletPartInd.empty())	_store.hvVerletPartInd.back() = nCollsions;	// for easier access
	ParallelFor(nParticles, [&](size_t i)
	{
		std::copy(_verletListCPU[i].begin(), _verletListCPU[i].end(), _store.hvVerletDst.begin() + _store.hvVerletPartInd[i]);
		std::fill(_store.hvVerletSrc.begin() + _store.hvVerletPartInd[i], _store.hvVerletSrc.begin() + _store.hvVerletPartInd[i] + _verletListCPU[i].size(), static_cast<unsigned>(i));
		if (m_Scene.m_PBC.bEnabled)
			std::copy(_verletListShiftsCPU[i].begin(), _verletListShiftsCPU[i].end(), _store.hvVirtShifts.begin() + _store.hvVerletPartInd[i]);
	});

	// update verlet lists on device
	m_gpu.UpdateVerletLists(_bPPVerlet, m_SceneGPU.GetPointerToParticles(), m_SceneGPU.GetPointerToWalls(), _store.hvVerletSrc, _store.hvVerletDst, _store.hvVerletPartInd,
		_store.hvVirtShifts, _collisions.vVerletSrc, _collisions.vVerletDst, _collisions.vVerletPartInd, _collisions.collisions);
	m_gpu.SortByDst(_bPPVerlet ? m_Scene.GetTotalParticlesNumber() : m_Scene.GetWallsNumber(),
		_collisions.vVerletSrc, _collisions.vVerletDst, _collisions.vVerletCollInd_DstSorted, _collisions.vVerletPartInd_DstSorted);
}

void CGPUSimulator::CUDAInitializeMaterials()
{
	if (m_pInteractProps)
	{
		CUDA_FREE_D(m_pInteractProps);
		m_pInteractProps = NULL;
	}
	size_t nCompounds = m_Scene.GetCompoundsNumber();
	m_gpu.SetCompoundsNumber( nCompounds );
	if (!nCompounds) return;

	SInteractProps* pInteractPropsHost;
	CUDA_MALLOC_H(&pInteractPropsHost, sizeof(SInteractProps)*nCompounds*nCompounds);
	CUDA_MALLOC_D(&m_pInteractProps,   sizeof(SInteractProps)*nCompounds*nCompounds);

	for (unsigned i = 0; i < nCompounds; ++i)
		for (unsigned j = 0; j < nCompounds; ++j)
			pInteractPropsHost[i*nCompounds + j] = m_Scene.GetInteractProp(i*nCompounds + j);

	CUDA_MEMCPY_H2D(m_pInteractProps, pInteractPropsHost, sizeof(SInteractProps)*nCompounds*nCompounds);
	CUDA_FREE_H(pInteractPropsHost);

}

void CGPUSimulator::CUDAInitializeWalls()
{
	std::vector<std::vector<unsigned>> vvWallsInGeom(m_pSystemStructure->GetGeometriesNumber());
	for (size_t i = 0; i < m_pSystemStructure->GetGeometriesNumber(); ++i)
	{
		const SGeometryObject* pGeom = m_pSystemStructure->GetGeometry(i);
		vvWallsInGeom[i].resize(pGeom->vPlanes.size());
		for (size_t j = 0; j < pGeom->vPlanes.size(); ++j)
			vvWallsInGeom[i][j] = static_cast<unsigned>(m_Scene.m_vNewIndexes[pGeom->vPlanes[j]]);
	}
	m_gpu.InitializeWalls(vvWallsInGeom, m_Scene.m_adjacentWalls);
}

void CGPUSimulator::MoveParticlesOverPBC()
{
	if (!m_Scene.m_PBC.bEnabled) return;
	m_gpu.MoveParticlesOverPBC(m_SceneGPU.GetPointerToParticles());
}
