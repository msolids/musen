/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BaseSimulator.h"
#include "GPUSimulator.cuh"
#include "SimplifiedSceneGPU.h"

class CGPUSimulator : public CBaseSimulator
{
	typedef std::vector<std::vector<unsigned>> std_matr_u;
	typedef std::vector<std::vector<uint8_t>> std_matr_u8;

	struct STempStorage
	{
		std_matr_u vDstSortedVerlet;
		thrust::host_vector<unsigned> hvVerletPartInd;
		thrust::host_vector<unsigned> hvVerletDst;
		thrust::host_vector<unsigned> hvVerletSrc;
		thrust::host_vector<uint8_t> hvVirtShifts;
	};

	struct SDispatchedResults
	{
		unsigned nActivePartNum;
		double dMaxSquaredPartDist;
		double dMaxWallVel;
	} *m_pDispatchedResults_d, *m_pDispatchedResults_h;

	CCUDADefines* m_cudaDefines{ nullptr };
	CGPU m_gpu;
	CSimplifiedSceneGPU m_SceneGPU;

	SInteractProps* m_pInteractProps;		// List of interaction properties according to current simplified scene.

public:
	CGPUSimulator();
	CGPUSimulator(const CBaseSimulator& _simulator);
	~CGPUSimulator();

	void SetExternalAccel(const CVector3& _accel) override;

	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	void GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID) override;

	CSimplifiedSceneGPU& GetPointerToSceneGPU() { return m_SceneGPU; }

	void StartSimulation() override;

	void InitializeModelParameters() override; // Sets model parameters to GPU memory.

private:
	void PerformSimulation();
	void LeapFrogStep(bool _bPredictionStep);
	void InitializeStep(double _dTimeStep) override;
	void CalculateForces(double _dTimeStep) override;
	void MoveObjects(double _dTimeStep, bool _bPredictionStep = false) override;

	void CalculateForcesPP(double _dTimeStep) override;
	void CalculateForcesPW(double _dTimeStep) override;
	void CalculateForcesSB(double _dTimeStep) override;
	void CalculateForcesEF(double _dTimeStep) override;

	void MoveParticles(bool _bPredictionStep);
	void MoveWalls(double _dTimeStep);

	void InitializeModels();
	void GenerateNewObjects();		// Generates new objects if necessary.
	void PrepareAdditionalSavingData() override;
	void SaveData() override;
	void UpdateVerletLists(double _dTimeStep);

	void CUDAUpdateGlobalCPUData();	// Dispatches necessary GPU operations and gathers data with one read.
	void CUDAUpdateActiveCollisions();
	void CUDAUpdateVerletLists(const std_matr_u& _verletListCPU, const std_matr_u8& _verletListShiftsCPU, CGPU::SCollisionsHolder& _collisions, STempStorage& _store, bool _bPPVerlet);
	void CUDAInitializeMaterials();

	void CUDAInitializeWalls();			// create or update information about walls for GPU

	void InitializeSimulator();

	// Check that all particles have correct coordinates and update coordinates of virtual particles.
	// If some real particles crossed the PBC boundaries, returns true (meaning the need to update verlet lists).
	void MoveParticlesOverPBC();
};

