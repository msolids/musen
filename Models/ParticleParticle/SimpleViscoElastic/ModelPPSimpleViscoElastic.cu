/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSimpleViscoElastic.cuh"
#include "ModelPPSimpleViscoElastic.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

__constant__ double m_vConstantModelParameters[2];

void CModelPPSimpleViscoElastic::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPSimpleViscoElastic::CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_VE_kernel,
		_particles.Vels,
		_particles.Forces,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.NormalOverlaps,
		_collisions.ContactVectors,

		_collisions.TotalForces
	);
}

void __global__ CUDA_CalcPPForce_VE_kernel(
	const CVector3 _partVels[],
	CVector3	   _partForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collNormalOverlaps[],
	const CVector3	_collContactVectors[],

	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned iColl       = _collActivityIndices[iActivColl];
		const unsigned iPart1      = _collSrcIDs[iColl];
		const unsigned iPart2      = _collDstIDs[iColl];
		const double   normOverlap = _collNormalOverlaps[iColl];

		// model parameters
		const double Kn = m_vConstantModelParameters[0];
		const double mu = m_vConstantModelParameters[1];

		const CVector3 normVector = _collContactVectors[iColl].Normalized();

		// relative velocity (normal)
		const CVector3 relVel      = _partVels[iPart2] - _partVels[iPart1];
		const double normRelVelLen = DotProduct(normVector, relVel);

		// normal force with damping
		const double normContactForceLen = -normOverlap * Kn;
		const double normDampingForceLen = mu * normRelVelLen;
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

		// store results in collision
		_collTotalForces[iColl] = normForce;

		// apply forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart1], normForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iPart2], normForce);
	}
}