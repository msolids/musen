/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBAerogel.cuh"
#include "ModelSBAerogel.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[7];
__constant__ SPBC PBC;

void CModelSBAerogel::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelSBAerogel::CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcSBForce_C1_kernel,
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
		_bonds.InitialLengths,
		_bonds.LeftIDs,
		_bonds.RightIDs,
		_bonds.NormalStiffnesses,
		_bonds.TangentialStiffnesses,

		_bonds.Activities,
		_bonds.EndActivities,
		_bonds.NormalPlasticStrains,
		_bonds.TangentialPlasticStrains,
		_bonds.NormalMoments,
		_bonds.PrevBonds,
		_bonds.TangentialMoments,
		_bonds.TangentialOverlaps,
		_bonds.TotalForces
	);
}

void __global__ CUDA_CalcSBForce_C1_kernel(
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
	const double	_bondInitialLengths[],
	const unsigned	_bondLeftIDs[],
	const unsigned	_bondRightIDs[],
	const double	_bondNormalStiffnesses[],
	const double	_bondTangentialStiffnesses[],

	bool		_bondActivities[],
	double		_bondEndActivities[],
	double		_bondNormalPlasticStrains[],
	CVector3	_bondTangentialPlasticStrains[],
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

		// relative angle velocity of contact partners
		CVector3 relAngleVel = _partAnglVels[_bondLeftIDs[i]] - _partAnglVels[_bondRightIDs[i]];

		// the bond in the global coordinate system
		CVector3 currentBond = GetSolidBond(_partCoords[_bondRightIDs[i]], _partCoords[_bondLeftIDs[i]], PBC);
		double dDistanceBetweenCenters = currentBond.Length();

		double dBondInitLength = _bondInitialLengths[i];

		// optimized
		CVector3 sumAngleVelocity = _partAnglVels[_bondLeftIDs[i]] + _partAnglVels[_bondRightIDs[i]];
		CVector3 relativeVelocity = _partVels[_bondLeftIDs[i]] - _partVels[_bondRightIDs[i]] - sumAngleVelocity * currentBond*0.5;

		CVector3 currentContact = currentBond / dDistanceBetweenCenters;
		CVector3 tempVector = _bondPrevBonds[i] * currentBond;

		CVector3 Phi = currentContact * (DotProduct(sumAngleVelocity, currentContact)*_timeStep*0.5);

		CMatrix3 M(1 + tempVector.z*Phi.z + tempVector.y*Phi.y, Phi.z - tempVector.z - tempVector.y*Phi.x, -Phi.y - tempVector.z*Phi.x + tempVector.y,
			tempVector.z - Phi.z - tempVector.x*Phi.y, tempVector.z*Phi.z + 1 + tempVector.x*Phi.x, -tempVector.z*Phi.y + Phi.x - tempVector.x,
			-tempVector.y - tempVector.x*Phi.z + Phi.y, -tempVector.y*Phi.z + tempVector.x - Phi.x, tempVector.y*Phi.y + tempVector.x*Phi.x + 1);

		CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
		CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

		// normal angle velocity
		CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);
		CVector3 tangAngleVel = relAngleVel - normalAngleVel;

		// calculate the force
		double &dBroken = _bondTangentialPlasticStrains[i].x;
		double dPureStrain = (dDistanceBetweenCenters - dBondInitLength) / dBondInitLength;
		double dPlasticStrainCompr = -fabs(m_vConstantModelParameters[0]);
		double dPlasticStrainTens  = fabs(m_vConstantModelParameters[1]);
		double dSoftnessRatioCompr = m_vConstantModelParameters[2];
		double dSoftnessRatioTens  = m_vConstantModelParameters[3];
		double dBreakageStrainTens = fabs(m_vConstantModelParameters[4]);
		double dHardnessRatioCompr = m_vConstantModelParameters[5];
		double dHardnessRatioTens  = m_vConstantModelParameters[6];
		double dKA = _bondCrossCuts[i] * _bondNormalStiffnesses[i];

		double dNormalActingForce = 0.;

		if (fabs(dBroken) < 0.5)
		{
			if ((dPureStrain < 0) && (dPureStrain < dPlasticStrainCompr))
			{
				double dDeltaPlasticStrain = (dPureStrain - dPlasticStrainCompr);
				dNormalActingForce = dPlasticStrainCompr * dKA + dDeltaPlasticStrain * dSoftnessRatioCompr*dKA;
				_bondNormalPlasticStrains[i] = dDeltaPlasticStrain * (1 - dSoftnessRatioCompr);
				if (dNormalActingForce > 0)
				{
					dBroken = -1.;
					dNormalActingForce = 0.;
				}
			}
			else if ((dPureStrain > 0) && (dPureStrain > dPlasticStrainTens))
			{
				double dDeltaPlasticStrain = (dPureStrain - dPlasticStrainTens);
				dNormalActingForce = dPlasticStrainTens * dKA + dDeltaPlasticStrain * dSoftnessRatioTens*dKA;
				_bondNormalPlasticStrains[i] = dDeltaPlasticStrain * (1 - dSoftnessRatioTens);
				if (dNormalActingForce < 0)
				{
					dBroken = 1.;
					dNormalActingForce = 0.;
				}
			}
			else
				dNormalActingForce = (dPureStrain - _bondNormalPlasticStrains[i]) * dKA;
			if (dNormalActingForce  > (dBreakageStrainTens*dKA)) // breakage by tension
			{
				_bondActivities[i] = false;
				_bondEndActivities[i] = _time;
			}
		}
		else if (dBroken < -0.5)
		{
			if (dPureStrain < _bondNormalPlasticStrains[i])
			{
				_bondNormalPlasticStrains[i] = dPureStrain;
			}
			dNormalActingForce = dKA * dHardnessRatioCompr * (dPureStrain - _bondNormalPlasticStrains[i]);
		}
		else if (dBroken > 0.5)
		{
			if (dPureStrain > _bondNormalPlasticStrains[i])
			{
				_bondNormalPlasticStrains[i] = dPureStrain;
			}
			dNormalActingForce = dKA * dHardnessRatioTens * (dPureStrain - _bondNormalPlasticStrains[i]);
		}

		CVector3 vNormalForce = -1 * currentContact * dNormalActingForce;

		_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
		const CVector3 vTangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * _bondCrossCuts[i] / dBondInitLength);
		const CVector3 vBondNormalMoment = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * _bondAxialMoments[i] * _bondTangentialStiffnesses[i] / dBondInitLength);
		const CVector3 vBondTangentialMoment = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * _bondNormalStiffnesses[i] * _bondAxialMoments[i] / dBondInitLength);

		const CVector3 vUnsymMoment = currentBond * 0.5 * vTangentialForce;

		_bondPrevBonds[i] = currentBond;
		_bondTotalForces[i] = vTangentialForce + vNormalForce;
		_bondNormalMoments[i] = vBondNormalMoment;
		_bondTangentialMoments[i] = vBondTangentialMoment;

		// apply forces and moments directly to particles
		const CVector3 partForce = vNormalForce + vTangentialForce;
		const CVector3 partMoment1 = vBondNormalMoment + vBondTangentialMoment - vUnsymMoment;
		const CVector3 partMoment2 = vBondNormalMoment + vBondTangentialMoment + vUnsymMoment;
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[_bondLeftIDs[i]], partForce);
		CUDA_VECTOR3_ATOMIC_ADD(_partMoments[_bondLeftIDs[i]], partMoment1);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[_bondRightIDs[i]], partForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partMoments[_bondRightIDs[i]], partMoment2);
	}
}