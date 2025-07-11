/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

//////////////////////////////////////////////////////////////////////////
/// GPU-version of the model.
/// The function is presented with the maximum possible set of parameters.
/// All unused parameters may be removed (here and and in ModelSB.cu) for speed-up.

void __global__ CUDA_CalcSBForce_kernel(
	double		_time,
	double		_timeStep,

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
	CVector3			_partForces[],
	CVector3			_partMoments[],

	unsigned			_bondsNum,
	const double		_bondAxialMoments[],
	const double		_bondCrossCuts[],
	const double		_bondDiameters[],
	const unsigned		_bondInitialIndices[],
	const double		_bondInitialLengths[],
	const unsigned		_bondLeftIDs[],
	const unsigned		_bondRightIDs[],
	const double		_bondNormalStiffnesses[],
	const double		_bondNormalStrengths[],
	const double		_bondTangentialStiffnesses[],
	const double		_bondTangentialStrengths[],
	const double		_bondTimeThermExpCoeffs[],
	const double		_bondViscosities[],
	const double		_bondYieldStrengths[],

	unsigned	_bondActivities[],
	double		_bondEndActivities[],
	CVector3	_bondNormalMoments[],
	double		_bondNormalPlasticStrains[],
	CVector3	_bondPrevBonds[],
	CVector3	_bondTangentialMoments[],
	CVector3	_bondTangentialOverlaps[],
	CVector3	_bondTangentialPlasticStrains[],
	CVector3	_bondTotalForces[]
);
