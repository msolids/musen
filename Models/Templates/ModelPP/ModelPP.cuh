/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

//////////////////////////////////////////////////////////////////////////
/// GPU-version of the model.
/// The function is presented with the maximum possible set of parameters.
/// All unused parameters may be removed (here and and in ModelPP.cu) for speed-up.

void __global__ CUDA_CalcPPForce_kernel(
	double					_time,
	double					_timeStep,
	const SInteractProps	_interactProps[],

	unsigned			_partsNum,
	const unsigned		_partActivities[],
	const CVector3		_partAnglVels[],
	const unsigned		_partCompoundIndices[],
	const double		_partContactRadii[],
	const CVector3		_partCoords[],
	const double		_partEndActivities[],
	const double		_partInertiaMoments[],
	const double		_partMasses[],
	const CQuaternion	_partQuaternions[],
	const double		_partRadii[],
	const CVector3		_partVels[],

	CVector3	_partForces[],
	CVector3	_partMoments[],

	unsigned			_collsNum,
	const unsigned*		_collActiveCollisionsNum,
	const bool			_collActivityFlags[],
	const unsigned		_collActivityIndices[],
	const uint16_t		_collInteractPropIDs[],
	const CVector3		_collContactVectors[],
	const unsigned		_collSrcIDs[],
	const unsigned		_collDstIDs[],
	const double		_collEquivMasses[],
	const double		_collEquivRadii[],
	const double		_collNormalOverlaps[],
	const uint8_t		_collVirtShifts[],

	CVector3 _collTangOverlaps[]
);
