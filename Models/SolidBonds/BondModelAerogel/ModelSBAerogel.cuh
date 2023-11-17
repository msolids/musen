/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcSBForce_C1_kernel(
	double	_time,
	double	_timeStep,

	const CVector3	_partAnglVels[],
	const CVector3	_partCoords[],
	const CVector3	_partVels[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	unsigned		_bondsNum,
	const double	_bondAxialMoments[],
	const double	_bondCrossCuts[],
	const double	_bondInitialLengths[],
	const unsigned	_bondLeftIDs[],
	const unsigned	_bondRightIDs[],
	const double	_bondNormalStiffnesses[],
	const double	_bondTangentialStiffnesses[],

	bool		_bondActivities[],
	double		_bondEndActivities[],
	double		_bondNormalPlasticStrains[],
	CVector3	_bondTangentialPlasticStrains[],
	CVector3	_bondNormalMoments[],
	CVector3	_bondPrevBonds[],
	CVector3	_bondTangentialMoments[],
	CVector3	_bondTangentialOverlaps[],
	CVector3	_bondTotalForces[]
);