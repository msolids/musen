/* Copyright (c) 2023, MUSEN Development Team.
   Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHeatTransfer.cuh"
#include "ModelPWHeatTransfer.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[3];
__constant__ SPBC PBC;

void CModelPWHeatTransfer::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWHeatTransfer::CalculatePWGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWHeatTransfer_kernel,
		_particles.Coords,
		_particles.ContactRadii,
		_particles.Temperatures,

		_particles.HeatFluxes,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.ContactVectors,  // interpreted as contact point
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.VirtualShifts
	);
}

void __global__ CUDA_CalcPWHeatTransfer_kernel(
	const CVector3	_partCoords[],
	const double	_partContactRadii[],
	const double    _partTemperatures[],

	double          _partHeatFluxes[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t   _collVirtShifts[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned iColl             = _collActivityIndices[iActivColl];
		const unsigned iPart             = _collDstIDs[iColl];
		const double   partContactRadius = _partContactRadii[iPart];
		const double   partTemperature   = _partTemperatures[iPart];

		const double wallTemperature   = m_vConstantModelParameters[0];
		const double heatTransferCoeff = m_vConstantModelParameters[1];
		const double resistivityFactor = m_vConstantModelParameters[2];

		const CVector3 rc = GPU_GET_VIRTUAL_COORDINATE(_partCoords[iPart]) - _collContactPoints[iColl];
		const double   rcLen = rc.Length();

		// normal overlap
		const double normOverlap = partContactRadius - rcLen;
		if (normOverlap < 0) continue;

		const double heatFlux = PI * partContactRadius * normOverlap * heatTransferCoeff * resistivityFactor * (wallTemperature - partTemperature);

		CUDA_ATOMIC_ADD(_partHeatFluxes[iPart], heatFlux);
	}
}