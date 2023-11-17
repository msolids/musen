/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWJKR.cuh"
#include "ModelPWJKR.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;

void CModelPWJKR::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWJKR::CalculatePWGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_JKR_kernel,
		_timeStep,
		_interactProps,

		_particles.AnglVels,
		_particles.Coords,
		_particles.Masses,
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
		_collisions.InteractPropIDs,
		_collisions.ContactVectors,  // interpreted as contact point
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.VirtualShifts,

		_collisions.TangOverlaps,
		_collisions.TotalForces
	);
}

void __global__ CUDA_CalcPWForce_JKR_kernel(
	double			     _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partAnglVels[],
	const CVector3	_partCoords[],
	const double	_partMasses[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	const CVector3	_wallVels[],
	const CVector3	_wallRotCenters[],
	const CVector3	_wallRotVels[],
	const CVector3  _wallNormalVecs[],
	CVector3        _wallForces[],

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
		const unsigned       iColl            = _collActivityIndices[iActivColl];
		const unsigned       iWall            = _collSrcIDs[iColl];
		const unsigned       iPart            = _collDstIDs[iColl];
		const SInteractProps prop             = _interactProps[_collInteractPropIDs[iColl]];
		const double         partRadius       = _partRadii[iPart];
		const CVector3       partAnglVel      = _partAnglVels[iPart];
		const CVector3       normVector       = _wallNormalVecs[iWall];
		const CVector3       tangOverlapOld   = _collTangOverlaps[iColl];

		const CVector3 rc     = GPU_GET_VIRTUAL_COORDINATE(_partCoords[iPart]) - _collContactPoints[iColl];
		const double   rcLen  = rc.Length();
		const CVector3 rcNorm = rc / rcLen;

		// normal overlap
		const double normOverlap = partRadius - rcLen;
		if (normOverlap < 0) continue;

		// normal and tangential relative velocity
		const CVector3 rotVel        = !_wallRotVels[iWall].IsZero() ? (_collContactPoints[iColl] - _wallRotCenters[iWall]) * _wallRotVels[iWall] : CVector3{ 0 };
		const CVector3 relVel        = _partVels[iPart] - _wallVels[iWall] + rotVel + rcNorm * partAnglVel * partRadius;
		const double   normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel    = normRelVelLen * normVector;
		const CVector3 tangRelVel    = relVel - normRelVel;

		// radius of the contact area
		const double contactAreaRadius = sqrt(partRadius * normOverlap);

		// normal force with damping
		const double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius;
		const double normContactForceLen = 4. * std::pow(contactAreaRadius, 3.) * prop.dEquivYoungModulus / (3. * partRadius) -
			std::sqrt(8. * PI * prop.dEquivYoungModulus * prop.dEquivSurfaceEnergy * std::pow(contactAreaRadius, 3.)) *
			std::abs(DotProduct(rcNorm, normVector));

		const double normDampingForceLen = _2_SQRT_5_6 * prop.dAlpha * normRelVelLen * sqrt(Kn * _partMasses[iPart]);
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

		// rotate old tangential overlap
		CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
		if (tangOverlapRot.IsSignificant())
			tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
		// calculate new tangential overlap
		CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

		// tangential force with damping
		const double Kt = 8 * prop.dEquivShearModulus * contactAreaRadius;
		const CVector3 tangShearForce = -Kt * tangOverlap;
		const CVector3 tangDampingForce = tangRelVel * (_2_SQRT_5_6 * prop.dAlpha * sqrt(Kt * _partMasses[iPart]));

		// check slipping condition and calculate total tangential force
		CVector3 tangForce;
		const double tangShearForceLen = tangShearForce.Length();
		const double frictionForceLen = prop.dSlidingFriction * fabs(normContactForceLen + normDampingForceLen);
		if (tangShearForceLen > frictionForceLen)
		{
			tangForce   = tangShearForce * frictionForceLen / tangShearForceLen;
			tangOverlap = tangForce / -Kt;
		}
		else
			tangForce   = tangShearForce + tangDampingForce;

		// rolling torque
		const CVector3 rollingTorque = partAnglVel.IsSignificant() ? partAnglVel * (-prop.dRollingFriction * fabs(normContactForceLen) * partRadius / partAnglVel.Length()) : CVector3{ 0 };

		// final forces and moments
		const CVector3 totalForce = normForce + tangForce;
		const CVector3 moment     = normVector * tangForce * -partRadius + rollingTorque;

		// store results in collision
		_collTangOverlaps[iColl] = tangOverlap;
		_collTotalForces[iColl]  = totalForce;

		// apply forces and moments
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart], moment);
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart], totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_wallForces[iWall], totalForce);
	}
}