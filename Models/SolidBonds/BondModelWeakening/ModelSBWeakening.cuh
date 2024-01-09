/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"
#include "curand_kernel.h"

void __global__ setup_kernel(curandState *state);

void __global__ CUDA_CalcSBForce_WK_kernel(
	double	_time,
	double	_timeStep,

	const CVector3	_partAnglVel[],
	const CVector3	_partCoord[],
	const CVector3	_partVel[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	unsigned		_bondsNum,
	const double	_bondAxialMoments[],
	const double	_bondCrossCuts[],
	const double	_bondDiameters[],
	const double	_bondInitialLengths[],
	const unsigned	_bondLeftIDs[],
	const unsigned	_bondRightIDs[],
	const double	_bondNormalStiffnesses[],
	const double	_bondNormalStrengths[],
	const double	_bondTangentialStiffnesses[],
	const double	_bondTangentialStrengths[],
	const double	_bondYieldStrengths[],

	curandState*	_randStates,
	uint8_t		_bondActivities[],
	double		_bondEndActivities[],
	CVector3	_bondNormalMoments[],
	double		_bondNormalPlasticStrains[],
	CVector3	_bondPrevBonds[],
	CVector3	_bondTangentialMoments[],
	CVector3	_bondTangentialOverlaps[],
	CVector3	_bondTangentialPlasticStrains[],
	CVector3	_bondTotalForces[]
);


 