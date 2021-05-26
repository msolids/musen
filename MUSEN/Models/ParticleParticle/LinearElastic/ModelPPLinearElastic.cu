/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPLinearElastic.cuh"
#include "ModelPPLinearElastic.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[2];

void CModelPPLinearElastic::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPLinearElastic::CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_LE_kernel,
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
		_collisions.NormalOverlaps,
		_collisions.ContactVectors,

		_collisions.TangOverlaps,
		_collisions.TotalForces
	);
}

void __global__ CUDA_CalcPPForce_LE_kernel(
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
	const double	_collNormalOverlaps[],
	const CVector3	_collContactVectors[],

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		double Kn = m_vConstantModelParameters[0];
		double Kt = m_vConstantModelParameters[1];

		const unsigned iColl      = _collActivityIndices[iActivColl];
		const unsigned iSrcPart   = _collSrcIDs[iColl];
		const unsigned iDstPart   = _collDstIDs[iColl];
		const SInteractProps prop = _interactProps[_collInteractPropIDs[iColl]];
		const double dEquivMass   = _collEquivMasses[iColl];
		const CVector3 srcAnglVel = _partAnglVels[iSrcPart];
		const CVector3 dstAnglVel = _partAnglVels[iDstPart];

		double dPartSrcRadius = _partRadii[iSrcPart];
		double dPartDstRadius = _partRadii[iDstPart];

		const CVector3 vContactVector = _collContactVectors[iColl];
		const CVector3 vRcSrc         = vContactVector * ( dPartSrcRadius / (dPartSrcRadius + dPartDstRadius));
		const CVector3 vRcDst         = vContactVector * (-dPartDstRadius / (dPartSrcRadius + dPartDstRadius));
		const CVector3 vNormalVector  = vContactVector.Normalized();

		// relative velocity (normal and tangential)
		const CVector3 vRelVel       = _partVels[iDstPart] + dstAnglVel * vRcDst - (_partVels[iSrcPart] + srcAnglVel * vRcSrc);
		const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
		const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
		const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

		// normal and tangential overlaps
		CVector3 vDeltaTangOverlap = vRelVelTang * _timeStep;

		// normal force with damping
		const double dDampingForce = -1.8257 * prop.dAlpha * dRelVelNormal * sqrt(Kn * dEquivMass);
		const double dNormalForce  = -_collNormalOverlaps[iColl] * Kn;

		// increment of tangential force with damping
		CVector3 vDampingTangForce = vRelVelTang * (-1.8257 * prop.dAlpha * sqrt(Kt * dEquivMass));

		// rotate old tangential force
		CVector3 vOldTangOverlap = _collTangOverlaps[iColl];
		CVector3 vTangOverlap = vOldTangOverlap - vNormalVector * DotProduct(vNormalVector, vOldTangOverlap);
		if (vTangOverlap.IsSignificant())
			vTangOverlap = vTangOverlap * vOldTangOverlap.Length() / vTangOverlap.Length();
		vTangOverlap += vDeltaTangOverlap;

		CVector3 vTangForce = vTangOverlap * Kt;

		// check slipping condition
		double dNewTangForce = vTangForce.Length();
		if (dNewTangForce > prop.dSlidingFriction * fabs(dNormalForce))
		{
			vTangForce = vTangForce * prop.dSlidingFriction * fabs(dNormalForce) / dNewTangForce;
			vTangOverlap = vTangForce / Kt;
		}
		else
			vTangForce += vDampingTangForce;

		// calculate rolling torque
		const CVector3 vRollingTorque1 = srcAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
			srcAnglVel * (-prop.dRollingFriction * fabs(dNormalForce) * dPartSrcRadius / srcAnglVel.Length()) : CVector3{ 0 };
		const CVector3 vRollingTorque2 = dstAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
			dstAnglVel * (-prop.dRollingFriction * fabs(dNormalForce) * dPartDstRadius / dstAnglVel.Length()) : CVector3{ 0 };

		// calculate moments and forces
		const CVector3 vTotalForce    = vNormalVector * (dNormalForce + dDampingForce) + vTangForce;
		const CVector3 vResultMoment1 = vNormalVector * vTangForce * dPartSrcRadius + vRollingTorque1;
		const CVector3 vResultMoment2 = vNormalVector * vTangForce * dPartDstRadius + vRollingTorque2;

		// store results in collision
		_collTangOverlaps[iColl] = vTangOverlap;
		_collTotalForces[iColl] = vTotalForce;

		// apply moments and forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iSrcPart], vTotalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iDstPart], vTotalForce);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iSrcPart], vResultMoment1);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iDstPart], vResultMoment2);
	}
}