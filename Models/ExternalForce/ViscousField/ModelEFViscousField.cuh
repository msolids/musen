/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BasicTypes.h"

void __global__ CUDA_CalcExtForce_VF_kernel(
	unsigned _partsNum,
	const double _partRadii[],
	const CVector3 _partVels[],
	CVector3 _partForces[]
);