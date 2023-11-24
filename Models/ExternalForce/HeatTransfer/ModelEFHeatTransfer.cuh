/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcExtForce_HT_kernel(
	unsigned      _partsNum,
	double       _time,
	const double _partRadii[],
	const double _partHeatCapacities[],
	const double _partTemperatures[],
	double _partHeatFluxes[]
);