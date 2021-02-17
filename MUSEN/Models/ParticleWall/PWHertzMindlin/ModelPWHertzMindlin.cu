/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHertzMindlin.cuh"
#include "ModelPWHertzMindlin.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;

void CModelPWHertzMindlin::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWHertzMindlin::CalculatePWForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_HM_kernel,
		_timeStep,
		_interactProps,

		_particles.AnglVels,
		_particles.Coords,
		_particles.Masses,
		_particles.Radii,
		_particles.Vels,
		_particles.Moments,

		_walls.Vels,
		_walls.RotCenters,
		_walls.RotVels,
		_walls.NormalVectors,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.ContactVectors,  // interpreted as Contact Point
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.VirtualShifts,

		_collisions.TangOverlaps,
		_collisions.TotalForces
	);
}

void __global__ CUDA_CalcPWForce_HM_kernel(
	double			     _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partAnglVels[],
	const CVector3	_partCoords[],
	const double	_partMasses[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	CVector3		_partMoments[],

	const CVector3	_wallVels[],
	const CVector3	_wallRotCenters[],
	const CVector3	_wallRotVels[],
	const CVector3  _wallNormalVecs[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t   _collVirtShifts[],

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned iColl       = _collActivityIndices[iActivColl];
		const unsigned iWall       = _collSrcIDs[iColl];
		const unsigned iPart       = _collDstIDs[iColl];
		const SInteractProps prop  = _interactProps[_collInteractPropIDs[iColl]];
		const double dPartRadius   = _partRadii[iPart];
		const CVector3 partAnglVel = _partAnglVels[iPart];

		const CVector3 vRc = GPU_GET_VIRTUAL_COORDINATE(_partCoords[iPart]) - _collContactPoints[iColl];
		const double   dRc = vRc.Length();
		const CVector3 nRc = vRc / dRc; // = vRc.Normalized()

		CVector3 vNormalVector = _wallNormalVecs[iWall];
		CVector3 vVelDueToRot  = !_wallRotVels[iWall].IsZero() ? (_collContactPoints[iColl] - _wallRotCenters[iWall]) * _wallRotVels[iWall] : CVector3{ 0 };

		// normal and tangential overlaps
		const double dNormalOverlap = dPartRadius - dRc;
		if (dNormalOverlap < 0) continue;

		// relative velocity (normal and tangential)
		const CVector3 vRelVel       = _partVels[iPart] - _wallVels[iWall] + vVelDueToRot + nRc * partAnglVel * dPartRadius;
		const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
		const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
		const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

		// normal force with damping
		const double Kn = 2 * prop.dEquivYoungModulus * sqrt(dPartRadius * dNormalOverlap);
		const double dDampingForce = 1.8257 * prop.dAlpha * dRelVelNormal * sqrt(Kn * _partMasses[iPart]);
		const double dNormalForce  = 2. / 3. * dNormalOverlap * Kn * fabs(DotProduct(nRc, vNormalVector));

		// increment of tangential force with damping
		const double Kt = 8 * prop.dEquivShearModulus * sqrt(dPartRadius * dNormalOverlap);
		const CVector3 vDampingTangForce = vRelVelTang * (1.8257 * prop.dAlpha * sqrt(Kt * _partMasses[iPart]));

		// rotate old tangential force
		CVector3 vTangOverlap = _collTangOverlaps[iColl] - vNormalVector * DotProduct(vNormalVector, _collTangOverlaps[iColl]);
		if (vTangOverlap.IsSignificant())
			vTangOverlap = vTangOverlap * _collTangOverlaps[iColl].Length() / vTangOverlap.Length();

		_collTangOverlaps[iColl] = vTangOverlap + vRelVelTang * _timeStep;
		CVector3 vTangForce = -Kt * _collTangOverlaps[iColl];
		// check slipping condition
		const double dNewTangForce = vTangForce.Length();
		if (dNewTangForce > prop.dSlidingFriction * fabs(dNormalForce))
		{
			vTangForce *= prop.dSlidingFriction * fabs(dNormalForce) / dNewTangForce;
			_collTangOverlaps[iColl] = vTangForce / -Kt;
		}
		else
			vTangForce += vDampingTangForce;

		// calculate rolling friction
		CVector3 vRollingTorque = partAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
			partAnglVel * (-prop.dRollingFriction * fabs(dNormalForce) * dPartRadius / partAnglVel.Length()) : CVector3{ 0 };

		// calculate and apply moment
		const CVector3 vMoment = vNormalVector * vTangForce * -dPartRadius + vRollingTorque;
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart], vMoment);

		// store results in collision
		_collTotalForces[iColl] = vTangForce + (dNormalForce + dDampingForce) * vNormalVector;
	}
}