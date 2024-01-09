/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcSBHeatConduction_kernel(
	const CVector3 _partCoords[],
	const double   _partTemperature[],
	double         _partHeatFlux[],

	unsigned       _bondsNum,
	uint8_t        _bondActivities[],
	const double   _bondCrossCuts[],
	const unsigned _bondLeftIDs[],
	const unsigned _bondRightIDs[],
	const double   _bondThermalConductivity[]
);


