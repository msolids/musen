/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSintering.cuh"
#include "ModelPPSintering.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[2];

void CModelPPSintering::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPSintering::CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_S_kernel,
		_interactProps,

		_particles.Vels,
		_particles.Forces,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps,
		_collisions.ContactVectors,

		_collisions.TotalForces
	);
}

void __global__ CUDA_CalcPPForce_S_kernel(
	const SInteractProps _interactProps[],

	const CVector3 _partVels[],
	CVector3       _partForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[],
	const CVector3  _collContactVectors[],

	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned iColl       = _collActivityIndices[iActivColl];
		const unsigned iPart1      = _collSrcIDs[iColl];
		const unsigned iPart2      = _collDstIDs[iColl];
		const double   equivRadius = _collEquivRadii[iColl];

		const CVector3 normVector = _collContactVectors[iColl].Normalized();

		// normal and tangential relative velocity
		CVector3 relVel     = _partVels[iPart1] - _partVels[iPart2];
		CVector3 normRelVel = normVector * DotProduct(normVector, relVel);
		CVector3 tangRelVel = relVel - normRelVel;

		// Bouvard and McMeeking's model
		const double squaredContactRadius = 4 * equivRadius * _collNormalOverlaps[iColl];

		// forces
		const CVector3 sinteringForce = normVector * 1.125 * PI * 2 * equivRadius * _interactProps[_collInteractPropIDs[iColl]].dEquivSurfaceEnergy;
		const CVector3 viscousForce   = normRelVel * (-PI * pow(squaredContactRadius, 2.0) / 8 / m_vConstantModelParameters[0]);
		const CVector3 tangForce      = tangRelVel * (-m_vConstantModelParameters[1] * PI * squaredContactRadius * pow(2 * equivRadius, 2.0) / 8 / m_vConstantModelParameters[0]);
		const CVector3 totalForce     = sinteringForce + viscousForce + tangForce;

		// store results in collision
		_collTotalForces[iColl] = totalForce;

		// apply forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart1], totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iPart2], totalForce);
	}
}