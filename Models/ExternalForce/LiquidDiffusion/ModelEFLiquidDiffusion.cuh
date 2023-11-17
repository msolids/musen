/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"
#include <curand_kernel.h>

void __global__ GenerateNewCurandStates_LiquidDiffusion_kernel(size_t _nStates, const unsigned* _seeds, curandState* _states);

void __global__ CUDA_CalcDifCoeff_LiquidDiffusion_kernel(
	double		        _timeStep,

	unsigned		    _partNum,
	const double		_partRadii[],
	const double		_partMasses[],
	const double		_intertiaMoments[],

	double				_pDragLinGPU[],
	double				_pFMagnitudeGPU[],
	double				_pDragRotGPU[],
	double				_pMMagnitudeGPU[],

	double				_pTauMinLin[],
	double				_pTauMinRot[]
);

void __global__ CUDA_CalcExtForce_LiquidDiffusion_kernel(
	unsigned			_partNum,
	const CVector3		_partVels[],
	const CVector3		_partAnglVels[],
	CVector3			_partForces[],
	CVector3			_partMoments[],
	const CVector3		_partCoords[],
	const double		_partRadii[],

	curandState*        _pCurandStatesGPU,

	const double		_pDragLinGPU[],
	const double		_pFMagnitudeGPU[],
	const double		_pDragRotGPU[],
	const double		_pMMagnitudeGPU[]
);

void __global__ CUDA_CalcKinEnergy_LiquidDiffusion_kernel(
	unsigned			_partNum,
	const CVector3		_partVels[],
	const CVector3		_partAnglVels[],

	const double		_partMasses[],
	const double		_intertiaMoments[],

	double				_partKinEnergies[]
);