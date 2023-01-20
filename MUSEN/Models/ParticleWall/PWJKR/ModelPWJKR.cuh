/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcPWForce_JKR_kernel(
	double			     _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partAnglVels[],
	const CVector3	_partCoords[],
	const double	_partMasses[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	CVector3		_partMoments[],

	const CVector3	_wallVels[],
	const CVector3	_wallRotCenters[],
	const CVector3	_wallRotVels[],
	const CVector3  _wallNormalVecs[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t   _collVirtShifts[],

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
);