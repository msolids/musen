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