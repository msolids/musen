/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcPWHeatTransfer_kernel(
	const CVector3	_partCoords[],
	const double	_partRadii[],
	const double    _partTemperatures[],

	double          _partHeatFluxes[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t   _collVirtShifts[]
);