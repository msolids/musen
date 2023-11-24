/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPChealNess.cuh"
#include "ModelPPChealNess.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[3];


void CModelPPChealNess::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPChealNess::CalculatePPGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_CN_kernel,
		_timeStep,
		_interactProps,

		_particles.Coords,
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

void __global__ CUDA_CalcPPForce_CN_kernel(
	double				 _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partCoords[],
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
		const unsigned       iColl          = _collActivityIndices[iActivColl];
		const unsigned       iPart1         = _collSrcIDs[iColl];
		const unsigned       iPart2         = _collDstIDs[iColl];
		const SInteractProps prop           = _interactProps[_collInteractPropIDs[iColl]];
		const double         equivMass      = _collEquivMasses[iColl];
		const CVector3       anglVel1       = _partAnglVels[iPart1];
		const CVector3       anglVel2       = _partAnglVels[iPart2];
		const double         radius1        = _partRadii[iPart1];
		const double         radius2        = _partRadii[iPart2];
		const CVector3       contactVector  = _collContactVectors[iColl];
		const CVector3       tangOverlapOld = _collTangOverlaps[iColl];

		// model parameters
		double minThickness  = m_vConstantModelParameters[0];
		double maxThickness  = m_vConstantModelParameters[1];
		double fluidVisosity = m_vConstantModelParameters[2];

		const CVector3 rc1        = contactVector * ( radius1 / (radius1 + radius2));
		const CVector3 rc2        = contactVector * (-radius2 / (radius1 + radius2));
		const CVector3 normVector = contactVector.Normalized();

		// adjusted normal overlap
		const double surfaceDistance = (_partCoords[iPart2] - _partCoords[iPart1]).Length() - radius1 - radius2;
		const double normOverlap = surfaceDistance < 0 ? fabs(surfaceDistance) : 0.0;

		// normal and tangential relative velocity
		const CVector3 relVel        = (_partVels[iPart2] + anglVel2 * rc2) - (_partVels[iPart1] + anglVel1 * rc1);
		const double   normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel    = normRelVelLen * normVector;
		const CVector3 tangRelVel    = relVel - normRelVel;

		// radius of the contact area
		const double contactAreaRadius = sqrt(_collEquivRadii[iColl] * normOverlap);

		// normal force with damping
		const double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius;
		const double normContactForceLen = -normOverlap * Kn * 2. / 3.;
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

		// calculate lubrication force
		CVector3 normLubrForce{ 0 };
		CVector3 tangLubrForce{ 0 };
		const double dh = contactVector.Length() - radius1 - radius2;
		if (minThickness <= dh && dh <= maxThickness)
		{
			double beta = radius2 / radius1;
			double Xi   = 2 * dh / (radius2 + radius1);
			double X11A = 6 * PI * radius1 * (2 * beta * beta / ((1 + pow(beta, 3.0)) * Xi) + beta * (1 + 7 * beta + beta * beta) / (5 * pow(1 + beta, 3.0)) * log(1 / Xi));
			double Y11A = 6 * PI * radius1 * (4 * beta * (1 + 7 * beta + beta * beta) / (15 * pow(1 + beta, 3.0)) * log(1 / Xi));
			double Y11B = -4 * PI * pow(radius1, 2.0) * (beta * (4 + beta) / (5 * pow(1 + beta, 2.0)) * log(1 / Xi));
			double Y11C = 8 * PI * pow(radius1, 3.0) * (2 * beta / (5 * (1 + beta)) * log(1 / Xi));
			double Y12C = Y11C / 4;
			double Y21B = -4 * PI * pow(radius2, 2.0) * ((4 + 1 / beta) / (5 * pow(1 + 1 / beta, 2.0)) * log(1 / Xi));
			CVector3 Nij       = -1 * normVector;
			CMatrix3 outerProd = OuterProduct(Nij, Nij);
			CVector3 velij     = _partVels[iPart1] - _partVels[iPart2];
			normLubrForce = -1 * fluidVisosity * ((X11A * outerProd + Y11A * (CMatrix3::Identity() - outerProd)) * velij + Y11B * (anglVel1 * Nij) + Y21B * (anglVel2 * Nij));
			tangLubrForce = -1 * fluidVisosity * (Y11B * velij * Nij - (CMatrix3::Identity() - outerProd) * (Y11C * anglVel1 + Y12C * anglVel2));
		}

		// final forces and moments
		const CVector3 totalForce    = normForce + tangForce + normLubrForce + tangLubrForce;
		const CVector3 resultMoment1 = normVector * (tangForce + tangLubrForce) * radius1 + rollingTorque1;
		const CVector3 resultMoment2 = normVector * (tangForce + tangLubrForce) * radius2 + rollingTorque2;

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