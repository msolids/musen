/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPJKR.cuh"
#include "ModelPPJKR.h"
#include <device_launch_parameters.h>

void CModelPPJKR::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
}

void CModelPPJKR::CalculatePPGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
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

		_collisions.TangOverlaps,
		_collisions.TotalForces
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

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned       iColl            = _collActivityIndices[iActivColl];
		const unsigned       iPart1           = _collSrcIDs[iColl];
		const unsigned       iPart2           = _collDstIDs[iColl];
		const SInteractProps prop             = _interactProps[_collInteractPropIDs[iColl]];
		const double         normOverlap      = _collNormalOverlaps[iColl];
		const double         equivMass        = _collEquivMasses[iColl];
		const double         equivRadius      = _collEquivRadii[iColl];
		const CVector3       anglVel1         = _partAnglVels[iPart1];
		const CVector3       anglVel2         = _partAnglVels[iPart2];
		const double         radius1          = _partRadii[iPart1];
		const double         radius2          = _partRadii[iPart2];
		const CVector3       contactVector    = _collContactVectors[iColl];
		const CVector3       tangOverlapOld   = _collTangOverlaps[iColl];

		const CVector3 rc1        = contactVector * (radius1 / (radius1 + radius2));
		const CVector3 rc2        = contactVector * (-radius2 / (radius1 + radius2));
		const CVector3 normVector = contactVector.Normalized();

		// normal and tangential relative velocity
		const CVector3 relVel        = (_partVels[iPart2] + anglVel2 * rc2) - (_partVels[iPart1] + anglVel1 * rc1);
		const double   normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel    = normRelVelLen * normVector;
		const CVector3 tangRelVel    = relVel - normRelVel;

		// radius of the contact area
		const double contactAreaRadius = sqrt(equivRadius * normOverlap);

		// normal force with damping
		const double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius;
		const double normContactForceLen = -1 * (4. * pow(contactAreaRadius, 3.) * prop.dEquivYoungModulus / (3. * equivRadius) -
			sqrt(8 * PI * prop.dEquivYoungModulus * prop.dEquivSurfaceEnergy * pow(contactAreaRadius, 3.)));
		const double normDampingForceLen = -_2_SQRT_5_6 * prop.dAlpha * normRelVelLen * sqrt(Kn * equivMass);
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

		// rotate old tangential overlap
		CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
		if (tangOverlapRot.IsSignificant())
			tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
		// calculate new tangential overlap
		CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

		// tangential force with damping
		const double Kt = 8 * prop.dEquivShearModulus * contactAreaRadius;
		const CVector3 tangShearForce = tangOverlap * Kt;
		const CVector3 tangDampingForce = tangRelVel * (-_2_SQRT_5_6 * prop.dAlpha * sqrt(Kt * equivMass));

		// check slipping condition and calculate total tangential force
		CVector3 tangForce;
		const double tangShearForceLen = tangShearForce.Length();
		const double frictionForceLen = prop.dSlidingFriction * fabs(normContactForceLen + normDampingForceLen);
		if (tangShearForceLen > frictionForceLen)
		{
			tangForce   = tangShearForce * frictionForceLen / tangShearForceLen;
			tangOverlap = tangForce / Kt;
		}
		else
			tangForce   = tangShearForce + tangDampingForce;

		// rolling torque
		const CVector3 rollingTorque1 = anglVel1.IsSignificant() ? anglVel1 * (-1 * prop.dRollingFriction * fabs(normContactForceLen) * radius1 / anglVel1.Length()) : CVector3{ 0 };
		const CVector3 rollingTorque2 = anglVel2.IsSignificant() ? anglVel2 * (-1 * prop.dRollingFriction * fabs(normContactForceLen) * radius2 / anglVel2.Length()) : CVector3{ 0 };

		// final forces and moments
		const CVector3 totalForce    = normForce + tangForce;
		const CVector3 resultMoment1 = normVector * tangForce * radius1 + rollingTorque1;
		const CVector3 resultMoment2 = normVector * tangForce * radius2 + rollingTorque2;

		// store results in collision
		_collTangOverlaps[iColl] = tangOverlap;
		_collTotalForces[iColl]  = totalForce;

		// apply moments and forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart1] , totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iPart2] , totalForce);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart1], resultMoment1);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart2], resultMoment2);
	}
}
