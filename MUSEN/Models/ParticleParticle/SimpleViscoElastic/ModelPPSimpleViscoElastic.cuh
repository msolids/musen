/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BasicTypes.h"

void __global__ CUDA_CalcPPForce_VE_kernel(
	const CVector3	_partVels[],
	CVector3		_partForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collNormalOverlaps[],
	const CVector3	_collContactVectors[],

	CVector3 _collTotalForces[]
);
