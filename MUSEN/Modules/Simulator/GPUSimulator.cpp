/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GPUSimulator.h"

CGPUSimulator::CGPUSimulator()
{
	Construct();
}

CGPUSimulator::CGPUSimulator(const CBaseSimulator& _other) :
	CBaseSimulator{ _other }
{
	Construct();
}

CGPUSimulator::~CGPUSimulator()
{
	if (m_pInteractProps)
		CUDA_FREE_D(m_pInteractProps);

	CUDA_FREE_D(m_pDispatchedResults_d);
	CUDA_FREE_H(m_pDispatchedResults_h);
}

void CGPUSimulator::Construct()
{
	CGPU::SetExternalAccel(m_externalAcceleration);
	m_gpu.SetAnisotropyFlag(m_considerAnisotropy);
	m_gpu.SetPBC(m_scene.m_PBC);

	CUDA_MALLOC_D(&m_pDispatchedResults_d, sizeof(SDispatchedResults));
	CUDA_MALLOC_H(&m_pDispatchedResults_h, sizeof(SDispatchedResults));
}

void CGPUSimulator::SetExternalAccel(const CVector3& _accel)
{
	CBaseSimulator::SetExternalAccel(_accel);
	CGPU::SetExternalAccel(m_externalAcceleration);
}

void CGPUSimulator::Initialize()
{
	CBaseSimulator::Initialize();

	CGPU::SetSimulationDomain(m_pSystemStructure->GetSimulationDomain());

	// get flag of anisotropy
	CGPU::SetAnisotropyFlag(m_considerAnisotropy);

	// Initialize scene, PBC, models on GPU
	m_sceneGPU.InitializeScene(m_scene, m_pSystemStructure);

	// store particle coordinates
	m_sceneGPU.CUDASaveVerletCoords();

	m_gpu.SetPBC(m_scene.m_PBC);

	CUDAInitializeWalls();
	m_gpu.InitializeCollisions();
	CUDAInitializeMaterials();
}

void CGPUSimulator::InitializeModelParameters()
{
	CBaseSimulator::InitializeModelParameters();
	for (auto& model : m_models)
		model->InitializeGPU(m_cudaDefines);
}

void CGPUSimulator::UpdateCollisionsStep(double _dTimeStep)
{
	// clear current states of particles and walls on GPU
	m_sceneGPU.ClearStates();

	// check that all particles are remains in simulation domain
	m_gpu.CheckParticlesInDomain(m_currentTime, m_sceneGPU.GetPointerToParticles(), &m_pDispatchedResults_d->nActivePartNum);

	// if there is no contact model, then there is no necessity to calculate contacts
	if (m_pPPModel || m_pPWModel)
	{
		UpdateVerletLists(_dTimeStep); // between PP and PW
		CUDAUpdateActiveCollisions();
	}
}

void CGPUSimulator::CalculateForcesStep(double _dTimeStep)
{
	if (m_pPPModel)   CalculateForcesPP(_dTimeStep);
	if (m_pPWModel)   CalculateForcesPW(_dTimeStep);
	if (m_pSBModel)   CalculateForcesSB(_dTimeStep);
	if (m_pLBModel)   CalculateForcesLB(_dTimeStep);
	if (m_pEFModel)   CalculateForcesEF(_dTimeStep);
	if (m_pPPHTModel) CalculateHeatTransferPP(_dTimeStep);

	cudaStreamQuery(0);
}

void CGPUSimulator::CalculateForcesPP(double _dTimeStep)
{
	if (!m_gpu.m_CollisionsPP.collisions.nElements) return;
	m_pPPModel->CalculatePPForceGPU(m_currentTime, _dTimeStep, m_pInteractProps, m_sceneGPU.GetPointerToParticles(), m_gpu.m_CollisionsPP.collisions);
}

void CGPUSimulator::CalculateForcesPW(double _dTimeStep)
{
	if (!m_gpu.m_CollisionsPW.collisions.nElements) return;
	m_pPWModel->CalculatePWForceGPU(m_currentTime, _dTimeStep, m_pInteractProps,
		m_sceneGPU.GetPointerToParticles(), m_sceneGPU.GetPointerToWalls(), m_gpu.m_CollisionsPW.collisions);
	m_gpu.GatherForcesFromPWCollisions(m_sceneGPU.GetPointerToParticles(), m_sceneGPU.GetPointerToWalls());
}

void CGPUSimulator::CalculateForcesSB(double _dTimeStep)
{
	if (m_scene.GetBondsNumber() == 0) return;
	m_pSBModel->CalculateSBForceGPU(m_currentTime, _dTimeStep, m_sceneGPU.GetPointerToParticles(), m_sceneGPU.GetPointerToSolidBonds());
}

void CGPUSimulator::CalculateForcesEF(double _dTimeStep)
{
	m_pEFModel->CalculateEFForceGPU(m_currentTime, _dTimeStep, m_sceneGPU.GetPointerToParticles());
}

void CGPUSimulator::CalculateHeatTransferPP(double _dTimeStep)
{
	m_pPPHTModel->CalculatePPHeatTransferGPU(m_currentTime, _dTimeStep, m_pInteractProps, m_sceneGPU.GetPointerToParticles(), m_gpu.m_CollisionsPP.collisions);
}

void CGPUSimulator::MoveParticles(bool _bPredictionStep)
{
	if (!m_externalAcceleration.IsZero())
		m_gpu.ApplyExternalAcceleration(m_sceneGPU.GetPointerToParticles());
	if (!_bPredictionStep)
	{
		if (m_variableTimeStep)
			m_currSimulationStep = m_gpu.CalculateNewTimeStep(m_currSimulationStep, m_initSimulationStep, m_partMoveLimit, m_timeStepFactor, m_sceneGPU.GetPointerToParticles());
		m_gpu.MoveParticles(m_currSimulationStep, m_initSimulationStep, m_sceneGPU.GetPointerToParticles(), m_variableTimeStep);
	}
	else
		m_gpu.MoveParticlesPrediction(m_currSimulationStep / 2., m_sceneGPU.GetPointerToParticles());

	MoveParticlesOverPBC(); // move virtual particles and check boundaries
}

void CGPUSimulator::MoveWalls(double _dTimeStep)
{
	m_wallsVelocityChanged = false;
	// analysis of transition to new interval
	for (unsigned iGeom = 0; iGeom < m_pSystemStructure->GeometriesNumber(); ++iGeom)
	{
		CRealGeometry* pGeom = m_pSystemStructure->Geometry(iGeom);

		if (pGeom->Planes().empty()) continue;
		if ((pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::FORCE_DEPENDENT) ||
			(pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::CONSTANT_FORCE)) // force
		{
			const CVector3 vTotalForce = m_gpu.CalculateTotalForceOnWall(iGeom, m_sceneGPU.GetPointerToWalls());
			pGeom->UpdateMotionInfo(vTotalForce.z);
		}
		else
			pGeom->UpdateMotionInfo(m_currentTime); // time

		CVector3 vVel = pGeom->GetCurrentVelocity();
		CVector3 vRotVel = pGeom->GetCurrentRotVelocity();
		CVector3 vRotCenter = pGeom->GetCurrentRotCenter();

		if (m_currentTime == 0 || vVel != pGeom->GetCurrentVelocity() || vRotVel != pGeom->GetCurrentRotVelocity() || vRotCenter != pGeom->GetCurrentRotCenter())
			m_wallsVelocityChanged = true;

		if ( !pGeom->FreeMotion().IsZero() )
			m_wallsVelocityChanged = true;

		if (vRotVel.IsZero() && pGeom->FreeMotion().IsZero() && vVel.IsZero()) continue;
		CMatrix3 RotMatrix;
		if (!vRotVel.IsZero())
			RotMatrix = CQuaternion(vRotVel*_dTimeStep).ToRotmat();

		m_gpu.MoveWalls(_dTimeStep, iGeom, vVel, vRotVel, vRotCenter, RotMatrix, pGeom->FreeMotion(),
			pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::FORCE_DEPENDENT, pGeom->RotateAroundCenter(), pGeom->Mass(), m_sceneGPU.GetPointerToWalls(), m_externalAcceleration);
	}
}

void CGPUSimulator::UpdateTemperatures(double _timeStep, bool _predictionStep)
{
	m_gpu.UpdateTemperatures(_timeStep, m_sceneGPU.GetPointerToParticles());
}

size_t CGPUSimulator::GenerateNewObjects()
{
	// TODO: find why contacts are not being detected. Check generation of agglomerates.
	//if (!m_generationManager->IsNeedToBeGenerated(m_currentTime)) return; // no need to generate new objects

	//// get actual data from device
	//m_sceneGPU.CUDAParticlesGPU2CPU(&m_scene);
	//m_sceneGPU.CUDABondsGPU2CPU(&m_scene);
	//m_sceneGPU.CUDAWallsGPU2CPU(&m_scene);
	//
	//const size_t newObjects = CBaseSimulator::GenerateNewObjects();
	//if (newObjects) // new objects have been generated
	//{
	//	// update data on device
	//	m_sceneGPU.CUDAParticlesCPU2GPU(&m_scene);
	//	m_sceneGPU.CUDABondsCPU2GPU(&m_scene);

	//	CUDAInitializeMaterials();
	//}
	return 0;
}

void CGPUSimulator::UpdatePBC()
{
	m_scene.m_PBC.UpdatePBC(m_currentTime);
	if (!m_scene.m_PBC.vVel.IsZero())	// if velocity is not zero
	{
		InitializeModelParameters();	// set new PBC to all models
		m_gpu.SetPBC(m_scene.m_PBC);	// set new PBC to GPU scene
	}
}

void CGPUSimulator::PrepareAdditionalSavingData()
{
	SParticleStruct& particles = m_scene.GetRefToParticles();
	SSolidBondStruct& bonds = m_scene.GetRefToSolidBonds();

	static SGPUCollisions PPCollisions(SBasicGPUStruct::EMemType::HOST), PWCollisions(SBasicGPUStruct::EMemType::HOST);
	m_gpu.CopyCollisionsGPU2CPU(PPCollisions, PWCollisions);

	// reset previously calculated stresses
	for (auto& data : m_additionalSavingData)
		data.stressTensor.Init(0);

	// save stresses caused by solid bonds
	for (size_t i = 0; i < bonds.Size(); ++i)
	{
		if (!bonds.Active(i) && !m_pSystemStructure->GetObjectByIndex(bonds.InitIndex(i))->IsActive(m_currentTime)) continue;
		const size_t leftID  = bonds.LeftID(i);
		const size_t rightID = bonds.RightID(i);
		CVector3 connVec = (particles.Coord(leftID) - particles.Coord(rightID)).Normalized();
		m_additionalSavingData[bonds.LeftID(i) ].AddStress(-1 * connVec * particles.Radius(leftID ),      bonds.TotalForce(i), PI * pow(2 * particles.Radius(leftID ), 3) / 6);
		m_additionalSavingData[bonds.RightID(i)].AddStress(     connVec * particles.Radius(rightID), -1 * bonds.TotalForce(i), PI * pow(2 * particles.Radius(rightID), 3) / 6);
	}

	// save stresses caused by particle-particle contact
	for (size_t i = 0; i < PPCollisions.nElements; ++i)
	{
		if (!PPCollisions.ActivityFlags[i]) continue;
		const size_t srcID = PPCollisions.SrcIDs[i];
		const size_t dstID = PPCollisions.DstIDs[i];
		CVector3 connVec = (particles.Coord(srcID) - particles.Coord(dstID)).Normalized();
		m_additionalSavingData[PPCollisions.SrcIDs[i]].AddStress(-1 * connVec * particles.Radius(srcID),      PPCollisions.TotalForces[i], PI * pow(2 * particles.Radius(srcID), 3) / 6);
		m_additionalSavingData[PPCollisions.DstIDs[i]].AddStress(     connVec * particles.Radius(dstID), -1 * PPCollisions.TotalForces[i], PI * pow(2 * particles.Radius(dstID), 3) / 6);
	}

	// save stresses caused by particle-wall contacts
	for (size_t i = 0; i < PWCollisions.nElements; ++i)
	{
		if (!PWCollisions.ActivityFlags[i]) continue;
		CVector3 connVec = (PWCollisions.ContactVectors[i] - particles.Coord(PWCollisions.DstIDs[i])).Normalized();
		m_additionalSavingData[PWCollisions.DstIDs[i]].AddStress(connVec * particles.Radius(PWCollisions.DstIDs[i]), PWCollisions.TotalForces[i], PI * pow(2 * particles.Radius(PWCollisions.DstIDs[i]), 3) / 6);
	}
}

void CGPUSimulator::SaveData()
{
	clock_t t = clock();
	cudaDeviceSynchronize();
	m_sceneGPU.CUDABondsGPU2CPU( m_scene );
	m_sceneGPU.CUDAParticlesGPU2CPUAllData(m_scene);
	m_sceneGPU.CUDAWallsGPU2CPUAllData(m_scene);
	m_nBrokenBonds = m_sceneGPU.GetBrokenBondsNumber();
	m_maxParticleVelocity = m_sceneGPU.GetMaxPartVelocity();
	if (m_scene.GetRefToParticles().ThermalsExist())
		m_maxParticleTemperature = m_sceneGPU.GetMaxPartTemperature();
	p_SaveData();

	m_verletList.AddDisregardingTimeInterval(clock() - t);
}

void CGPUSimulator::UpdateVerletLists(double _dTimeStep)
{
	CUDAUpdateGlobalCPUData();
	if (m_verletList.IsNeedToBeUpdated(_dTimeStep, sqrt(m_pDispatchedResults_h->dMaxSquaredPartDist) , m_maxWallVelocity))
	{
		m_sceneGPU.CUDAParticlesGPU2CPUVerletData(m_scene);
		m_sceneGPU.CUDAWallsGPU2CPUVerletData(m_scene);
		m_verletList.UpdateList(m_currentTime);

		static STempStorage storePP, storePW; // to reuse memory
		CUDAUpdateVerletLists(m_verletList.m_PPList, m_verletList.m_PPVirtShift, m_gpu.m_CollisionsPP, storePP, true);
		CUDAUpdateVerletLists(m_verletList.m_PWList, m_verletList.m_PWVirtShift, m_gpu.m_CollisionsPW, storePW, false);
		m_sceneGPU.CUDASaveVerletCoords();
	}
}

void CGPUSimulator::CUDAUpdateGlobalCPUData()
{
	// update max velocities
	m_sceneGPU.GetMaxSquaredPartDist(&m_pDispatchedResults_d->dMaxSquaredPartDist);
	if (m_wallsVelocityChanged)
		m_sceneGPU.GetMaxWallVelocity(&m_pDispatchedResults_d->dMaxWallVel);

	CUDA_MEMCPY_D2H(m_pDispatchedResults_h, m_pDispatchedResults_d, sizeof(SDispatchedResults));

	const bool bNewInactiveParticles = (m_nInactiveParticles != m_sceneGPU.GetParticlesNumber() - m_pDispatchedResults_h->nActivePartNum);
	if (bNewInactiveParticles && m_sceneGPU.GetBondsNumber())
	{
		m_gpu.CheckBondsActivity(m_currentTime, m_sceneGPU.GetPointerToParticles(), m_sceneGPU.GetPointerToSolidBonds());
		m_sceneGPU.CUDABondsActivityGPU2CPU(m_scene);
		m_scene.UpdateParticlesToBonds();
	}

	m_nInactiveParticles = m_sceneGPU.GetParticlesNumber() - m_pDispatchedResults_h->nActivePartNum;
	if (m_wallsVelocityChanged)
		m_maxWallVelocity = m_pDispatchedResults_h->dMaxWallVel;
}

void CGPUSimulator::CUDAUpdateActiveCollisions()
{
	m_gpu.UpdateActiveCollisionsPP(m_sceneGPU.GetPointerToParticles());
	m_gpu.UpdateActiveCollisionsPW(m_sceneGPU.GetPointerToParticles(), m_sceneGPU.GetPointerToWalls());
}

void CGPUSimulator::CUDAUpdateVerletLists(const std_matr_u& _verletListCPU, const std_matr_u8& _verletListShiftsCPU,
	CGPU::SCollisionsHolder& _collisions, STempStorage& _store, bool _bPPVerlet)
{
	const size_t nParticles = m_scene.GetTotalParticlesNumber();
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
		if (m_scene.m_PBC.bEnabled)
			std::copy(_verletListShiftsCPU[i].begin(), _verletListShiftsCPU[i].end(), _store.hvVirtShifts.begin() + _store.hvVerletPartInd[i]);
	});

	// update verlet lists on device
	m_gpu.UpdateVerletLists(_bPPVerlet, m_sceneGPU.GetPointerToParticles(), m_sceneGPU.GetPointerToWalls(), _store.hvVerletSrc, _store.hvVerletDst, _store.hvVerletPartInd,
		_store.hvVirtShifts, _collisions.vVerletSrc, _collisions.vVerletDst, _collisions.vVerletPartInd, _collisions.collisions);
	m_gpu.SortByDst(_bPPVerlet ? m_scene.GetTotalParticlesNumber() : m_scene.GetWallsNumber(),
		_collisions.vVerletSrc, _collisions.vVerletDst, _collisions.vVerletCollInd_DstSorted, _collisions.vVerletPartInd_DstSorted);
}

void CGPUSimulator::CUDAInitializeMaterials()
{
	if (m_pInteractProps)
	{
		CUDA_FREE_D(m_pInteractProps);
		m_pInteractProps = NULL;
	}
	size_t nCompounds = m_scene.GetCompoundsNumber();
	m_gpu.SetCompoundsNumber( nCompounds );
	if (!nCompounds) return;

	SInteractProps* pInteractPropsHost;
	CUDA_MALLOC_H(&pInteractPropsHost, sizeof(SInteractProps)*nCompounds*nCompounds);
	CUDA_MALLOC_D(&m_pInteractProps,   sizeof(SInteractProps)*nCompounds*nCompounds);

	for (unsigned i = 0; i < nCompounds; ++i)
		for (unsigned j = 0; j < nCompounds; ++j)
			pInteractPropsHost[i*nCompounds + j] = m_scene.GetInteractProp(i*nCompounds + j);

	CUDA_MEMCPY_H2D(m_pInteractProps, pInteractPropsHost, sizeof(SInteractProps)*nCompounds*nCompounds);
	CUDA_FREE_H(pInteractPropsHost);

}

void CGPUSimulator::CUDAInitializeWalls()
{
	std::vector<std::vector<unsigned>> vvWallsInGeom(m_pSystemStructure->GeometriesNumber());
	for (size_t i = 0; i < m_pSystemStructure->GeometriesNumber(); ++i)
	{
		const CRealGeometry* pGeom = m_pSystemStructure->Geometry(i);
		const auto& planes = pGeom->Planes();
		vvWallsInGeom[i].resize(planes.size());
		for (size_t j = 0; j < planes.size(); ++j)
			vvWallsInGeom[i][j] = static_cast<unsigned>(m_scene.m_vNewIndexes[planes[j]]);
	}
	m_gpu.InitializeWalls(vvWallsInGeom, m_scene.m_adjacentWalls);
}

void CGPUSimulator::MoveParticlesOverPBC()
{
	if (!m_scene.m_PBC.bEnabled) return;
	m_gpu.MoveParticlesOverPBC(m_sceneGPU.GetPointerToParticles());
}

void CGPUSimulator::GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID)
{
	m_gpu.GetOverlapsInfo(m_sceneGPU.GetPointerToParticles(), _nMaxParticleID, _dMaxOverlap, _dAverageOverlap);
}
