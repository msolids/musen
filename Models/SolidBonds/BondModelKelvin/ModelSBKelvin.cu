/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBKelvin.cuh"
#include "ModelSBKelvin.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[1];
__constant__ SPBC PBC;

void CModelSBKelvin::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelSBKelvin::CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcSBForce_Kelvin_kernel,
		_time,
		_timeStep,

		_particles.AnglVels,
		_particles.Coords,
		_particles.Vels,
		_particles.Forces,
		_particles.Moments,

		static_cast<unsigned>(_bonds.nElements),
		_bonds.AxialMoments,
		_bonds.CrossCuts,
		_bonds.Diameters,
		_bonds.InitialLengths,
		_bonds.LeftIDs,
		_bonds.RightIDs,
		_bonds.Viscosities,
		_bonds.NormalStiffnesses,
		_bonds.NormalStrengths,
		_bonds.TangentialStiffnesses,
		_bonds.TangentialStrengths,

		_bonds.Activities,
		_bonds.EndActivities,
		_bonds.NormalMoments,
		_bonds.PrevBonds,
		_bonds.TangentialMoments,
		_bonds.TangentialOverlaps,
		_bonds.TotalForces
	);
}

void __global__ CUDA_CalcSBForce_Kelvin_kernel(
	double	_time,
	double	_timeStep,

	const CVector3	_partAnglVels[],
	const CVector3	_partCoords[],
	const CVector3	_partVels[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	unsigned		_bondsNum,
	const double	_bondAxialMoments[],
	const double	_bondCrossCuts[],
	const double	_bondDiameters[],
	const double	_bondInitialLengths[],
	const unsigned	_bondLeftIDs[],
	const unsigned	_bondRightIDs[],
	const double	_bondViscosities[],
	const double	_bondNormalStiffnesses[],
	const double	_bondNormalStrengths[],
	const double	_bondTangentialStiffnesses[],
	const double	_bondTangentialStrengths[],

	uint8_t		_bondActivities[],
	double		_bondEndActivities[],
	CVector3	_bondNormalMoments[],
	CVector3	_bondPrevBonds[],
	CVector3	_bondTangentialMoments[],
	CVector3	_bondTangentialOverlaps[],
	CVector3	_bondTotalForces[]
)
{
	for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _bondsNum; i += blockDim.x * gridDim.x)
	{
		if (!_bondActivities[i]) continue;

		// model parameters
		const bool considerBreakage = m_vConstantModelParameters[0] != 0.0;
		const double& maxDampRatio = m_vConstantModelParameters[1];

		// relative angle velocity of contact partners
		const CVector3 relAngleVel = _partAnglVels[_bondLeftIDs[i]] - _partAnglVels[_bondRightIDs[i]];

		// the bond in the global coordinate system
		const CVector3 currentBond = GetSolidBond(_partCoords[_bondRightIDs[i]], _partCoords[_bondLeftIDs[i]], PBC);
		double distanceBetweenCenters = currentBond.Length();

		// optimized
		const CVector3 sumAngleVelocity = _partAnglVels[_bondLeftIDs[i]] + _partAnglVels[_bondRightIDs[i]];
		const CVector3 relativeVelocity = _partVels[_bondLeftIDs[i]] - _partVels[_bondRightIDs[i]] - sumAngleVelocity * currentBond * 0.5;

		const CVector3 currentContact = currentBond / distanceBetweenCenters;
		const CVector3 tempVector = _bondPrevBonds[i] * currentBond;

		const CVector3 phi = currentContact * (DotProduct(sumAngleVelocity, currentContact) * _timeStep * 0.5);

		const CMatrix3 M(1 + tempVector.z * phi.z + tempVector.y * phi.y, phi.z - tempVector.z - tempVector.y * phi.x, -phi.y - tempVector.z * phi.x + tempVector.y,
			tempVector.z - phi.z - tempVector.x * phi.y, tempVector.z * phi.z + 1 + tempVector.x * phi.x, -tempVector.z * phi.y + phi.x - tempVector.x,
			-tempVector.y - tempVector.x * phi.z + phi.y, -tempVector.y * phi.z + tempVector.x - phi.x, tempVector.y * phi.y + tempVector.x * phi.x + 1);

		const CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
		const CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

		// normal angle velocity
		const CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);
		const CVector3 tangAngleVel = relAngleVel - normalAngleVel;

		// calculate the force
		const double strainTotal = (distanceBetweenCenters - _bondInitialLengths[i]) / _bondInitialLengths[i];
		const CVector3 normalForce = currentContact * (-1 * _bondCrossCuts[i] * _bondNormalStiffnesses[i] * strainTotal);

		CVector3 dampingForceNorm = -_bondViscosities[i] * normalVelocity * _bondCrossCuts[i] * _bondNormalStiffnesses[i] * fabs(strainTotal);
		if (maxDampRatio != 0.0)
		{
			if (dampingForceNorm.Length() > maxDampRatio * normalForce.Length())
				dampingForceNorm *= maxDampRatio * normalForce.Length() / dampingForceNorm.Length();
		}

		_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
		const CVector3 tangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * _bondCrossCuts[i] / _bondInitialLengths[i]);

		CVector3 dampingForceTang = -_bondViscosities[i] * tangentialVelocity * _bondTangentialOverlaps[i].Length() * (_bondTangentialStiffnesses[i] * _bondCrossCuts[i] / _bondInitialLengths[i]);
		if (maxDampRatio != 0.0)
		{
			if (dampingForceTang.Length() > maxDampRatio * tangentialForce.Length())
				dampingForceTang *= maxDampRatio * tangentialForce.Length() / dampingForceTang.Length();
		}

		const CVector3 bondNormalMoment = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * _bondAxialMoments[i] * _bondTangentialStiffnesses[i] / _bondInitialLengths[i]);
		const CVector3 bondTangentialMoment = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * _bondNormalStiffnesses[i] * _bondAxialMoments[i] / _bondInitialLengths[i]);
		const CVector3 bondUnsymMoment = currentBond * 0.5 * tangentialForce;

		_bondNormalMoments[i] = bondNormalMoment;
		_bondTangentialMoments[i] = bondTangentialMoment;
		_bondPrevBonds[i] = currentBond;
		_bondTotalForces[i] = normalForce + tangentialForce + dampingForceNorm + dampingForceTang;

		if (considerBreakage)
		{
			double forceLength = normalForce.Length();
			if (strainTotal <= 0)	// compression
				forceLength *= -1;

			// check the bond destruction
			const double maxStress = forceLength / _bondCrossCuts[i] + bondTangentialMoment.Length() * _bondDiameters[i] / (2 * _bondAxialMoments[i]);
			const double maxTorque = tangentialForce.Length() / _bondCrossCuts[i] + bondNormalMoment.Length() * _bondDiameters[i] / (4 * _bondAxialMoments[i]);

			if (maxStress >= _bondNormalStrengths[i] || maxTorque >= _bondTangentialStrengths[i])
			{
				_bondActivities[i] = false;
				_bondEndActivities[i] = _time;
				continue; // if bond is broken do not apply forces and moments
			}
		}

		// apply forces and moments directly to particles, only if bond is not broken
		const CVector3 partForce = normalForce + tangentialForce + dampingForceNorm + dampingForceTang;
		const CVector3 partMoment1 = bondNormalMoment + bondTangentialMoment - bondUnsymMoment;
		const CVector3 partMoment2 = bondNormalMoment + bondTangentialMoment + bondUnsymMoment;
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[_bondLeftIDs[i]], partForce);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[_bondLeftIDs[i]], partMoment1);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[_bondRightIDs[i]], partForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partMoments[_bondRightIDs[i]], partMoment2);
	}
}