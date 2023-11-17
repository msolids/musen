/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BasicTypes.h"

void __global__ CUDA_CalcPPForce_HMVDW_kernel(
	double				 _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partAnglVels[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	const CVector3	_partCoords[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivMasses[],
	const double	_collEquivRadii[],
	const CVector3	_collContactVectors[],

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
);
