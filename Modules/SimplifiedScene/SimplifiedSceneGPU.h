/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SimplifiedSceneGPU.cuh"
#include "SceneTypesGPU.h"
#include "SimplifiedScene.h"

class CSimplifiedSceneGPU
{
	CGPUScene m_gpuScene;

protected:
	SGPUSolidBonds m_SolidBonds;
	SGPUParticles m_Particles;
	SGPUWalls m_Walls;

public:
	CSimplifiedSceneGPU(const CCUDADefines* _cudaDefines);

	void InitializeScene(CSimplifiedScene& _Scene, CSystemStructure* _pSystemStructure);

	inline size_t GetBondsNumber() const { return m_SolidBonds.nElements; }
	inline size_t GetParticlesNumber() const { return m_Particles.nElements; }
	inline size_t GetWallsNumber() const { return m_Walls.nElements; }

	inline SGPUSolidBonds& GetPointerToSolidBonds() { return m_SolidBonds; }
	inline SGPUParticles& GetPointerToParticles() { return m_Particles; }
	inline SGPUWalls& GetPointerToWalls() { return m_Walls; }

	void ClearStates() const; // Sets current values of running variables (force, moment, heat flux) to 0.
	void GetMaxSquaredPartDist(double* _bufMaxVelocity);
	double GetMaxPartVelocity();
	double GetMaxPartTemperature();
	void GetMaxWallVelocity(double* _bufMaxVelocity);
	size_t GetInactiveBondsNumber() const;
	void GetActiveBondsNumber(unsigned* _bufNumber) const;

	void CUDABondsCPU2GPU(CSimplifiedScene& _pSceneCPU);				// update info about bonds on GPU
	void CUDABondsGPU2CPU(CSimplifiedScene& _pSceneCPU);				// update info about bonds on CPU
	void CUDABondsGPU2CPUDynamicData(CSimplifiedScene& _sceneCPU) const; // update info about bonds on CPU, which change with time
	void CUDAParticlesCPU2GPU(CSimplifiedScene& _pSceneCPU);			// update info about particles on CPU
	void CUDAParticlesGPU2CPUVerletData(CSimplifiedScene& _pSceneCPU);	// update info about particles on CPU, needed for verlet lists
	void CUDAParticlesGPU2CPUDynamicData(CSimplifiedScene& _sceneCPU) const; // update info about particles on CPU, which change with time
	void CUDAParticlesGPU2CPUAllData(CSimplifiedScene& _pSceneCPU);		// update info about particles on CPU
	void CUDABondsActivityGPU2CPU(CSimplifiedScene& _pSceneCPU);		// update info about bonds activity on CPU
	void CUDAWallsCPU2GPU(CSimplifiedScene& _pSceneCPU);				// update info about walls on GPU
	void CUDAWallsGPU2CPUVerletData(CSimplifiedScene& _pSceneCPU);		// update info about walls on CPU, needed for verlet lists
	void CUDAWallsGPU2CPUAllData(CSimplifiedScene& _pSceneCPU);			// update info about walls on CPU
	void CUDASaveVerletCoords() const;
};


