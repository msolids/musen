/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcExtForce_CS_kernel(
	unsigned _partsNum,

	const double   _partRadii[],
	const CVector3 _partCoords[],
	const CVector3 _partVels[],
	const double   _partMasses[],

	CVector3 _partForces[]
);