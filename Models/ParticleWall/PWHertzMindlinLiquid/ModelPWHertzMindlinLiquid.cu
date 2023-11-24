/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHertzMindlinLiquid.cuh"
#include "ModelPWHertzMindlinLiquid.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;
__constant__ double m_vConstantModelParameters[4];

void CModelPWHertzMindlinLiquid::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelPWHertzMindlinLiquid::CalculatePWGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_HML_kernel,
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

void __global__ CUDA_CalcPWForce_HML_kernel(
	double		         _timeStep,
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
	const CVector3	_wallNormalVecs[],
	CVector3        _wallForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const CVector3	_collContactPoints[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const uint8_t	_collVirtShifts[],

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		// model parameters
		const double minThickness   = m_vConstantModelParameters[0];
		const double contactAngle   = m_vConstantModelParameters[1] * PI / 180;
		const double surfaceTension = m_vConstantModelParameters[2];
		const double viscosity      = m_vConstantModelParameters[3];

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

		const double bondLength = fmax(rcLen, minThickness);

		// normal overlap
		const double normOverlap = partRadius - rcLen;

		// normal and tangential relative velocity
		const CVector3 rotVel        = !_wallRotVels[iWall].IsZero() ? (_collContactPoints[iColl] - _wallRotCenters[iWall]) * _wallRotVels[iWall] : CVector3{ 0 };
		const CVector3 relVel        = _partVels[iPart] - _wallVels[iWall] + rotVel + rcNorm * partAnglVel * partRadius;
		const double   normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel    = normRelVelLen * normVector;
		const CVector3 tangRelVel    = relVel - normRelVel;

		// wet contact
		const double bondVolume = PI / 3. * pow(partRadius, 3.0); // one quarter of summarized sphere volume
		const double A = -1.1 * pow(bondVolume, -0.53);
		const double tempLn = log(bondVolume);
		const double B = (-0.34 * tempLn - 0.96) * contactAngle * contactAngle - 0.019 * tempLn + 0.48;
		const double C = 0.0042 * tempLn + 0.078;
		const CVector3 capForce = normVector * PI * partRadius * surfaceTension * (exp(A * bondLength + B) + C);
		const CVector3 normViscForce = normRelVel * -1 * (6 * PI * viscosity * partRadius * partRadius / bondLength);
		const CVector3 tangForceLiq = tangRelVel * (6 * PI * viscosity * partRadius * (8.0 / 15.0 * log(partRadius / bondLength) + 0.9588));
		const CVector3 momentLiq = rc * tangForceLiq;

		CVector3 normForceDry{ 0 }, tangForceDry{ 0 }, tangShearForceDry{ 0 }, tangDampingForceDry{ 0 }, rollingTorqueDry{ 0 }, tangOverlap{ 0 };
		if (normOverlap >= 0)
		{
			// contact radius
			const double contactRadius = sqrt(partRadius * normOverlap);

			// normal force with damping
			const double Kn = 2 * prop.dEquivYoungModulus * contactRadius;
			const double normContactForceDryLen = 2. / 3. * normOverlap * Kn * fabs(DotProduct(rcNorm, normVector));
			const double normDampingForceDryLen = _2_SQRT_5_6 * prop.dAlpha * normRelVelLen * sqrt(Kn * _partMasses[iPart]);
			normForceDry = normVector * (normContactForceDryLen + normDampingForceDryLen);

			// rotate old tangential overlap
			CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
			if (tangOverlapRot.IsSignificant())
				tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
			// calculate new tangential overlap
			tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

			// tangential force with damping
			const double Kt = 8 * prop.dEquivShearModulus * contactRadius;
			tangShearForceDry = -Kt * tangOverlap;
			tangDampingForceDry = tangRelVel * (_2_SQRT_5_6 * prop.dAlpha * sqrt(Kt * _partMasses[iPart]));

			// check slipping condition and calculate total tangential force
			const double tangShearForceDryLen = tangShearForceDry.Length();
			const double frictionForceLen = prop.dSlidingFriction * fabs(normContactForceDryLen + normDampingForceDryLen);
			if (tangShearForceDryLen > frictionForceLen)
			{
				tangForceDry = tangShearForceDry * frictionForceLen / tangShearForceDryLen;
				tangOverlap  = tangForceDry / -Kt;
			}
			else
				tangForceDry = tangShearForceDry + tangDampingForceDry;

			// rolling torque
			if (partAnglVel.IsSignificant())
				rollingTorqueDry = partAnglVel * (-prop.dRollingFriction * fabs(normContactForceDryLen) * partRadius / partAnglVel.Length());
		}

		// final forces and moments
		const CVector3 tangForce  = tangForceDry + tangForceLiq;
		const CVector3 totalForce = normForceDry + tangForce + capForce + normViscForce;
		const CVector3 moment1    = normVector * tangForce * -partRadius + rollingTorqueDry - momentLiq;

		// store results in collision
		_collTangOverlaps[iColl] = tangOverlap;
		_collTotalForces[iColl]  = totalForce;

		// apply forces and moments
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart], moment1);
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart], totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_wallForces[iWall], totalForce);
	}
}
