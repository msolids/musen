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
		_collisions.ContactVectors
	);
}

void __global__ CUDA_CalcPPForce_S_kernel(
	const SInteractProps	_interactProps[],
	const CVector3	_partVels[],
	CVector3	_partForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[],
	const CVector3  _collContactVectors[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned iColl = _collActivityIndices[iActivColl];
		const unsigned iSrcPart = _collSrcIDs[iColl];
		const unsigned iDstPart = _collDstIDs[iColl];
		const double dEquivRadius = _collEquivRadii[iColl];

		CVector3 vNormalVector = _collContactVectors[iColl].Normalized();

		//obtain velocities
		CVector3 vRelVel           = _partVels[iSrcPart] - _partVels[iDstPart];
		CVector3 vRelVelNormal     = vNormalVector * DotProduct(vNormalVector, vRelVel);
		CVector3 vRelVelTangential = vRelVel - vRelVelNormal;

		//Bouvard and McMeeking's model
		const double dSquaredContactRadius = 4 * dEquivRadius * _collNormalOverlaps[iColl];

		// calculate forces
		const CVector3 vSinteringForce = vNormalVector * 1.125 * PI * 2 * dEquivRadius * _interactProps[_collInteractPropIDs[iColl]].dEquivSurfaceEnergy;
		const CVector3 vViscousForce = vRelVelNormal * (-PI * pow(dSquaredContactRadius, 2) / 8 / m_vConstantModelParameters[0]);
		const CVector3 vTangentialForce = vRelVelTangential * (-m_vConstantModelParameters[1] * PI * dSquaredContactRadius * pow(2 * dEquivRadius, 2) / 8 / m_vConstantModelParameters[0]);
		const CVector3 vTotalForce = vSinteringForce + vViscousForce + vTangentialForce;

		// apply forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iSrcPart], vTotalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iDstPart], vTotalForce);
	}
}