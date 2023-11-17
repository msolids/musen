/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcPWForce_VE_kernel(
	const CVector3	_partCoords[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	const CVector3	_wallVels[],
	const CVector3	_wallRotCenters[],
	const CVector3	_wallRotVels[],
	const CVector3	_wallNormalVecs[],
	CVector3		_wallForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t   _collVirtShifts[],

	CVector3 _collTotalForces[]
);