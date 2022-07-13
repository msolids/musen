#pragma once

#include "BasicTypes.h"
void __global__ CUDA_CalcPPForce_S_Temp_kernel(
	double               _time,
	double	             _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partVels[],
	const double	_partTemperatures[],
	CVector3		_partForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivMasses[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[],
	const CVector3  _collContactVectors[],

	double	 _collInitNormalOverlaps[],
	CVector3 _collTangOverlaps[]
);
