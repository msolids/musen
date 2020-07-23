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

void CModelPWHertzMindlinLiquid::CalculatePWForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_HML_kernel,
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

void __global__ CUDA_CalcPWForce_HML_kernel(
	double		         _timeStep,
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
	const CVector3	_wallNormalVecs[],

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
		double dMinThickness   = m_vConstantModelParameters[0];
		double dContactAngle   = m_vConstantModelParameters[1] * PI / 180;
		double dSurfaceTension = m_vConstantModelParameters[2];
		double dViscosity      = m_vConstantModelParameters[3];

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

		double dBondLength = dRc;
		if (dBondLength < dMinThickness)
			dBondLength = dMinThickness;

		// relative velocity (normal and tangential)
		const CVector3 vRelVel       = _partVels[iPart] - _wallVels[iWall] + vVelDueToRot + nRc * partAnglVel * dPartRadius;
		const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
		const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
		const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

		// wet contact
		double dBondVolume = PI / 3. * pow(dPartRadius, 3); // one quarter of summarized sphere volume
		double dA = -1.1 * pow(dBondVolume, -0.53);
		double dTempLn = log(dBondVolume);
		double dB = (-0.34 * dTempLn - 0.96) * dContactAngle * dContactAngle - 0.019 * dTempLn + 0.48;
		double dC = 0.0042 * dTempLn + 0.078;
		CVector3 dCapForce = vNormalVector * PI * dPartRadius * dSurfaceTension * (exp(dA * dBondLength + dB) + dC);
		CVector3 dViscForceNormal = vRelVelNormal * -1 * (6 * PI * dViscosity * dPartRadius * dPartRadius / dBondLength);
		CVector3 vTangForceLiquid = vRelVelTang * (6 * PI * dViscosity * dPartRadius * (8.0 / 15.0 * log(dPartRadius / dBondLength) + 0.9588));
		CVector3 vMomentLiquid = vRc * vTangForceLiquid;

		// normal and Tangential overlaps
		const double dNormalOverlap = dPartRadius - dRc;

		CVector3 vNormalForce(0), vTangForceDry(0), vDampingTangForceDry(0), vRollingTorqueDry(0);
		if (dNormalOverlap < 0)
		{
			_collTangOverlaps[iColl].Init(0);
		}
		else
		{
			// normal force with damping
			const double Kn = 2 * prop.dEquivYoungModulus * sqrt(dPartRadius * dNormalOverlap);
			const double dDampingNormalForceDry =  1.8257 * prop.dAlpha * dRelVelNormal * sqrt(Kn * _partMasses[iPart]);
			const double dNormalForceDry =  2. / 3. * dNormalOverlap * Kn * fabs(DotProduct(nRc, vNormalVector));

			// increment of tangential force with damping
			double Kt = 8 * prop.dEquivShearModulus * sqrt(dPartRadius * dNormalOverlap);
			vDampingTangForceDry = vRelVelTang * (1.8257 * prop.dAlpha * sqrt(Kt * _partMasses[iPart]));

			// rotate old tangential force
			CVector3 vTangOverlap = _collTangOverlaps[iColl] - vNormalVector * DotProduct(vNormalVector, _collTangOverlaps[iColl]);
			if (vTangOverlap.IsSignificant())
				vTangOverlap = vTangOverlap * _collTangOverlaps[iColl].Length() / vTangOverlap.Length();

			_collTangOverlaps[iColl] = vTangOverlap + vRelVelTang * _timeStep;
			vTangForceDry = _collTangOverlaps[iColl] * -Kt;
			// check slipping condition
			double dNewTangForce = vTangForceDry.Length();
			if (dNewTangForce > prop.dSlidingFriction * fabs(dNormalForceDry))
			{
				vTangForceDry = vTangForceDry * (prop.dSlidingFriction * fabs(dNormalForceDry) / dNewTangForce);
				_collTangOverlaps[iColl] = vTangForceDry / -Kt;
			}

			// calculate Rolling friction
			if (partAnglVel.IsSignificant())
				vRollingTorqueDry = partAnglVel * (-prop.dRollingFriction * fabs(dNormalForceDry) * dPartRadius / partAnglVel.Length());

			vNormalForce = vNormalVector * (dNormalForceDry + dDampingNormalForceDry);
		}

		// save old tangential force
		CVector3 vTangForce = vTangForceDry + vDampingTangForceDry + vTangForceLiquid;

		// calculate and apply moment
		const CVector3 vMoment = vNormalVector * vTangForce * -dPartRadius + vRollingTorqueDry - vMomentLiquid;
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[iPart], vMoment);

		// add result to the arrays
		_collTotalForces[iColl] = vTangForce + vNormalForce + dCapForce + dViscForceNormal;
	}
}
