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
		unsigned activeBondsNumBeforeBreak;
		unsigned activeBondsNumAfterBreak;
		double dMaxSquaredPartDist;
		double dMaxWallVel;
	};

	CCUDADefines* m_cudaDefines{ new CCUDADefines{} };
	CGPU m_gpu{ m_cudaDefines };
	CSimplifiedSceneGPU m_sceneGPU{ m_cudaDefines };
	SInteractProps* m_pInteractProps{ nullptr };		// List of interaction properties according to current simplified scene.
	SDispatchedResults* m_pDispatchedResults_d{ nullptr };
	SDispatchedResults* m_pDispatchedResults_h{ nullptr };

public:
	CGPUSimulator();
	explicit CGPUSimulator(const CBaseSimulator& _other);
	CGPUSimulator(const CGPUSimulator& _other) = delete;
	CGPUSimulator(CGPUSimulator&& _other) = delete;
	CGPUSimulator& operator=(const CGPUSimulator& _other) = delete;
	CGPUSimulator& operator=(CGPUSimulator&& _other) = delete;
	~CGPUSimulator() override;

	void SetExternalAccel(const CVector3& _accel) override;

	CSimplifiedSceneGPU& GetPointerToSceneGPU() { return m_sceneGPU; }

	void Initialize() override;
	void InitializeModelParameters() override; // Sets model parameters to GPU memory.

	void UpdateCollisionsStep(double _dTimeStep) override;

	void CalculateForcesStep(double _dTimeStep) override;
	void CalculateForcesPP(double _dTimeStep) override;
	void CalculateForcesPW(double _dTimeStep) override;
	void CalculateForcesSB(double _dTimeStep) override;
	void CalculateForcesEF(double _dTimeStep) override;

	void MoveParticles(bool _bPredictionStep = false) override;
	void MoveWalls(double _dTimeStep) override;
	void UpdateTemperatures(bool _predictionStep = false) override;

	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	void GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID) override;

private:
	void GenerateNewObjects() override;	// Generates new objects if necessary.
	void UpdatePBC() override;				// Updates moving PBC.

	void PrepareAdditionalSavingData() override;
	void SaveData() override;
	void UpdateVerletLists(double _dTimeStep);

	void CUDAUpdateGlobalCPUData();	// Dispatches necessary GPU operations and gathers data with one read.
	void CUDAUpdateActiveCollisions();
	void CUDAUpdateVerletLists(const std_matr_u& _verletListCPU, const std_matr_u8& _verletListShiftsCPU, CGPU::SCollisionsHolder& _collisions, STempStorage& _store, bool _bPPVerlet);
	void CUDAInitializeMaterials();

	void CUDAInitializeWalls();			// create or update information about walls for GPU

	void Construct(); // Constructs the simulator.

	// Check that all particles have correct coordinates and update coordinates of virtual particles.
	// If some real particles crossed the PBC boundaries, returns true (meaning the need to update verlet lists).
	void MoveParticlesOverPBC();
};

