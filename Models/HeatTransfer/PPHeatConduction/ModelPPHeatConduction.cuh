#pragma once

#include "BasicTypes.h"

void __global__ CUDA_CalcPPHeatTransfer_HC_kernel(
	const double	_partTemperatures[],

	double _partHeatFluxes[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[]
);
