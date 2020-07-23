/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPJKR.cuh"
#include "ModelPPJKR.h"
#include <device_launch_parameters.h>

void CModelPPJKR::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
}

void CModelPPJKR::CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_JKR_kernel,
		_timeStep,
		_interactProps,

		_particles.AnglVels,
		_particles.Radii,
		_particles.Vels,
		_particles.Forces,
		_particles.Moments,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivMasses,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps,
		_collisions.ContactVectors,

		_collisions.TangOverlaps
	);
}

void __global__ CUDA_CalcPPForce_JKR_kernel(
	double					_timeStep,
	const SInteractProps	_interactProps[],

	const CVector3	_partAnglVels[],
	const double	_partRadii[],
	const CVector3	_partVels[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivMasses[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[],
	const CVector3	_collContactVectors[],

	CVector3 _collTangOverlaps[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		unsigned iColl            = _collActivityIndices[iActivColl];
		unsigned iSrcPart         = _collSrcIDs[iColl];
		unsigned iDstPart         = _collDstIDs[iColl];
		SInteractProps prop       = _interactProps[_collInteractPropIDs[iColl]];
		double dEquivMass         = _collEquivMasses[iColl];
		const CVector3 srcAnglVel = _partAnglVels[iSrcPart];
		const CVector3 dstAnglVel = _partAnglVels[iDstPart];

		double dPartSrcRadius = _partRadii[iSrcPart];
		double dPartDstRadius = _partRadii[iDstPart];

		const CVector3 vContactVector = _collContactVectors[iColl];
		const CVector3 vRcSrc         = vContactVector * (dPartSrcRadius / (dPartSrcRadius + dPartDstRadius));
		const CVector3 vRcDst         = vContactVector * (-dPartDstRadius / (dPartSrcRadius + dPartDstRadius));
		const CVector3 vNormalVector  = vContactVector.Normalized();

		// relative velocity (normal and tangential)
		const CVector3 vRelVel       = _partVels[iDstPart] - _partVels[iSrcPart] + vRcSrc * srcAnglVel - vRcDst * dstAnglVel;
		const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
		const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
		const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

		// normal and tangential overlaps
		double dContactRadius = sqrt(_collNormalOverlaps[iColl] * _collEquivRadii[iColl]);

		// normal force with damping
		double Kn = 2 * prop.dEquivYoungModulus * dContactRadius;
		double dNormalForce = -1 * (4. * pow(dContactRadius, 3.) * prop.dEquivYoungModulus / (3. * _collEquivRadii[iColl]) -
			sqrt(8 * PI * prop.dEquivYoungModulus * prop.dEquivSurfaceEnergy * pow(dContactRadius, 3.)));

		const double dDampingForce = (-1)*1.8257*prop.dAlpha*dRelVelNormal*sqrt(Kn*dEquivMass);

		// increment of tangential force with damping
		double Kt = 8 * prop.dEquivShearModulus*dContactRadius;
		CVector3 vDampingTangForce = vRelVelTang * (-1.8257 * prop.dAlpha * sqrt(Kt * dEquivMass));

		// rotate old tangential force
		CVector3 vOldTangOverlap = _collTangOverlaps[iColl];
		CVector3 vTangOverlap = vOldTangOverlap - vNormalVector * DotProduct(vNormalVector, vOldTangOverlap);
		if (vTangOverlap.IsSignificant())
			vTangOverlap = vTangOverlap * vOldTangOverlap.Length() / vTangOverlap.Length();
		vTangOverlap += vRelVelTang * _timeStep;

		CVector3 vTangForce = vTangOverlap * Kt;

		// check slipping condition
		double dNewTangForce = vTangForce.Length();
		if (dNewTangForce > prop.dSlidingFriction * fabs(dNormalForce))
		{
			vTangForce = vTangForce * (prop.dSlidingFriction * fabs(dNormalForce) / dNewTangForce);
			vTangOverlap = vTangForce / Kt;
		}
		// save tangential force
		vTangForce += vDampingTangForce;

		// calculate rolling torque
		const CVector3 vRollingTorque1 = srcAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
			srcAnglVel * (-prop.dRollingFriction * fabs(dNormalForce) * dPartSrcRadius / srcAnglVel.Length()) : CVector3{ 0 };
		const CVector3 vRollingTorque2 = dstAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
			dstAnglVel * (-prop.dRollingFriction * fabs(dNormalForce) * dPartDstRadius / dstAnglVel.Length()) : CVector3{ 0 };

		// calculate moments and forces
		_collTangOverlaps[iColl] = vTangOverlap;
		const CVector3 vTotalForce    = vNormalVector * (dNormalForce + dDampingForce) + vTangForce;
		const CVector3 vResultMoment1 = vNormalVector * vTangForce * dPartSrcRadius + vRollingTorque1;
		const CVector3 vResultMoment2 = vNormalVector * vTangForce * dPartDstRadius + vRollingTorque2;

		// apply moments and forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iSrcPart], vTotalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iDstPart], vTotalForce);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iSrcPart], vResultMoment1);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iDstPart], vResultMoment2);
	}
}
