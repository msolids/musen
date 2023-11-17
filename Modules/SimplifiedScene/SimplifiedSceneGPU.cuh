/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BasicGPUFunctions.cuh"
#include "SceneTypesGPU.h"

class CGPUScene
{
	const CCUDADefines* m_cudaDefines{ nullptr };

public:
	void SetCudaDefines(const CCUDADefines* _cudaDefines);

	void GetMaxSquaredPartVerletDistance(SGPUParticles& _particles, double* _bufMaxVel);
	double GetMaxPartVelocity(SGPUParticles& _particles) const;
	double GetMaxPartTemperature(SGPUParticles& _particles) const;
	void GetMaxWallVelocity(SGPUWalls& _walls, double* _bufMaxVel) const;

	size_t GetBrokenBondsNumber(const SGPUSolidBonds& _bonds) const;
};