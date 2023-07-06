/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWSimpleViscoelastic.cuh"
#include "ModelPWSimpleViscoElastic.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;
__constant__ double m_vConstantModelParameters[2];

void CModelPWSimpleViscoElastic::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWSimpleViscoElastic::CalculatePWForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_VE_kernel,
		_particles.Coords,
		_particles.Radii,
		_particles.Vels,
		_particles.Forces,
		_particles.Moments,

		_walls.Vels,
		_walls.RotCenters,
		_walls.RotVels,
		_walls.NormalVectors,
		_walls.Forces,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.ContactVectors,  // interpreted as Contact Point
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.VirtualShifts,

		_collisions.TotalForces
	);
}

void __global__ CUDA_CalcPWForce_VE_kernel(
	const CVector3	_partCoords[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	const CVector3	_wallVels[],
	const CVector3	_wallRotCenters[],
	const CVector3	_wallRotVels[],
	const CVector3	_wallNormalVecs[],
	CVector3        _wallForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t   _collVirtShifts[],

	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned iColl      = _collActivityIndices[iActivColl];
		const unsigned iWall      = _collSrcIDs[iColl];
		const unsigned iPart      = _collDstIDs[iColl];
		const CVector3 normVector = _wallNormalVecs[iWall];

		// model parameters
		const double Kn = m_vConstantModelParameters[0];
		const double mu = m_vConstantModelParameters[1];

		const CVector3 rc     = GPU_GET_VIRTUAL_COORDINATE(_partCoords[iPart]) - _collContactPoints[iColl];
		const double   rcLen  = rc.Length();
		const CVector3 rcNorm = rc / rcLen;

		// normal overlap
		const double normOverlap = _partRadii[iPart] - rcLen;
		if (normOverlap < 0) continue;

		// normal and tangential relative velocity
		const CVector3 rotVel   = !_wallRotVels[iWall].IsZero() ? (_collContactPoints[iColl] - _wallRotCenters[iWall]) * _wallRotVels[iWall] : CVector3{ 0 };
		const CVector3 relVel   = _partVels[iPart] - _wallVels[iWall] + rotVel;
		const double normRelVel = DotProduct(normVector, relVel);

		// normal force with damping
		const double normContactForceLen = normOverlap * Kn * fabs(DotProduct(rcNorm, normVector));
		const double normDampingForceLen = -mu * normRelVel;
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

		// store results in collision
		_collTotalForces[iColl] = normForce;

		// apply forces and moments
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart], normForce);
		CUDA_VECTOR3_ATOMIC_SUB(_wallForces[iWall], normForce);
	}
}