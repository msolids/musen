/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBPlasticConcrete.cuh"
#include "ModelSBPlasticConcrete.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[2];
__constant__ SPBC PBC;

void CModelSBPlasticConcrete::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelSBPlasticConcrete::CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcSBForce_PC_kernel,
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
		_bonds.NormalStiffnesses,
		_bonds.NormalStrengths,
		_bonds.TangentialStiffnesses,
		_bonds.TangentialStrengths,
		_bonds.YieldStrengths,

		_bonds.Activities,
		_bonds.EndActivities,
		_bonds.NormalMoments,
		_bonds.NormalPlasticStrains,
		_bonds.PrevBonds,
		_bonds.TangentialMoments,
		_bonds.TangentialOverlaps,
		_bonds.TangentialPlasticStrains,
		_bonds.TotalForces
	);
}

__global__ void CUDA_CalcSBForce_PC_kernel(
	double	_time,
	double	_timeStep,

	const CVector3	_partAnglVel[],
	const CVector3	_partCoord[],
	const CVector3	_partVel[],
	CVector3		_partForces[],
	CVector3		_partMoments[],

	unsigned		_bondsNum,
	const double	_bondAxialMoments[],
	const double	_bondCrossCuts[],
	const double	_bondDiameters[],
	const double	_bondInitialLengths[],
	const unsigned	_bondLeftIDs[],
	const unsigned	_bondRightIDs[],
	const double	_bondNormalStiffnesses[],
	const double	_bondNormalStrengths[],
	const double	_bondTangentialStiffnesses[],
	const double	_bondTangentialStrengths[],
	const double	_bondYieldStrengths[],

	unsigned	_bondActivities[],
	double		_bondEndActivities[],
	CVector3	_bondNormalMoments[],
	double		_bondNormalPlasticStrains[],
	CVector3	_bondPrevBonds[],
	CVector3	_bondTangentialMoments[],
	CVector3	_bondTangentialOverlaps[],
	CVector3	_bondTangentialPlasticStrains[],
	CVector3	_bondTotalForces[]
)
{
	for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _bondsNum; i += blockDim.x * gridDim.x)
	{
		if (!_bondActivities[i]) continue;

		// relative angle velocity of contact partners
		CVector3 relAngleVel = _partAnglVel[_bondLeftIDs[i]] - _partAnglVel[_bondRightIDs[i]];

		// the bond in the global coordinate system
		CVector3 currentBond = GetSolidBond(_partCoord[_bondRightIDs[i]], _partCoord[_bondLeftIDs[i]], PBC);
		double dDistanceBetweenCenters = currentBond.Length();
		CVector3 rAC = currentBond * 0.5;

		double dDruckYieldStrength = -_bondYieldStrengths[i];
		double dZugYieldStrength = 0.7 * _bondYieldStrengths[i];
		double dBetta = 0.5;
		double dAlpha = 0.3;
		double dKn = _bondNormalStiffnesses[i];

		// optimized
		CVector3 sumAngleVelocity = _partAnglVel[_bondLeftIDs[i]] + _partAnglVel[_bondRightIDs[i]];
		CVector3 relativeVelocity = _partVel[_bondLeftIDs[i]] - _partVel[_bondRightIDs[i]] - sumAngleVelocity* rAC;

		CVector3 currentContact = currentBond / dDistanceBetweenCenters;
		CVector3 tempVector = _bondPrevBonds[i] * currentBond;

		CVector3 Phi = currentContact*(DotProduct(sumAngleVelocity, currentContact)*_timeStep*0.5);

		CMatrix3 M(	1 + tempVector.z*Phi.z + tempVector.y*Phi.y,	Phi.z - tempVector.z - tempVector.y*Phi.x,		-Phi.y - tempVector.z*Phi.x + tempVector.y,
					tempVector.z - Phi.z - tempVector.x*Phi.y,		tempVector.z*Phi.z + 1 + tempVector.x*Phi.x,	-tempVector.z*Phi.y + Phi.x - tempVector.x,
					-tempVector.y - tempVector.x*Phi.z + Phi.y,		-tempVector.y*Phi.z + tempVector.x - Phi.x,		tempVector.y*Phi.y + tempVector.x*Phi.x + 1);

		CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
		CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

		// normal angle velocity
		CVector3 normalAngleVel = currentContact*DotProduct(currentContact, relAngleVel);
		CVector3 tangAngleVel = relAngleVel - normalAngleVel;

		// calculate the force
		double dStrainTotal = (dDistanceBetweenCenters - _bondInitialLengths[i]) / _bondInitialLengths[i];
		double dElasticStrain = dStrainTotal - _bondNormalPlasticStrains[i];
		double dCurrentNormalStress;
		if (dElasticStrain >= 0) //tension
		{
			dCurrentNormalStress = dElasticStrain * _bondNormalStiffnesses[i];
			double dLimitStress = dZugYieldStrength - _bondNormalPlasticStrains[i] * dKn * dAlpha;
			if (dCurrentNormalStress > dLimitStress)
			{
				_bondNormalPlasticStrains[i] += (dCurrentNormalStress - dLimitStress) * (1 + dAlpha) / dKn;
				dElasticStrain = dStrainTotal - _bondNormalPlasticStrains[i];
				dCurrentNormalStress = dElasticStrain * _bondNormalStiffnesses[i];
			}
			if (m_vConstantModelParameters[0] != 0.0 && dCurrentNormalStress < 0) // bond breakage
			{
				_bondActivities[i] = false;
				_bondEndActivities[i] = _time;
			}
		}
		else // druck
		{
			dCurrentNormalStress = dElasticStrain * _bondNormalStiffnesses[i];
			double dLimitStress = dDruckYieldStrength + _bondNormalPlasticStrains[i] * dKn * dBetta;
			if (dCurrentNormalStress < dLimitStress)
			{
				_bondNormalPlasticStrains[i] += (dCurrentNormalStress - dLimitStress) * (1 - dBetta) / dKn;
				dElasticStrain = dStrainTotal - _bondNormalPlasticStrains[i];
				dCurrentNormalStress = dElasticStrain * _bondNormalStiffnesses[i];
			}
		}
		CVector3 vNormalForce = currentContact*(dCurrentNormalStress*_bondCrossCuts[i]);

		_bondTangentialOverlaps[i] = M*_bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
		_bondTangentialPlasticStrains[i] = M*_bondTangentialPlasticStrains[i];
		CVector3 vTangStress = (_bondTangentialOverlaps[i] - _bondTangentialPlasticStrains[i])*_bondTangentialStiffnesses[i];

		CVector3 vTangentialForce = vTangStress * _bondCrossCuts[i];
		_bondNormalMoments[i] = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * _bondAxialMoments[i] * _bondTangentialStiffnesses[i]) / _bondInitialLengths[i];
		_bondTangentialMoments[i] = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * _bondNormalStiffnesses[i] * _bondAxialMoments[i]) / _bondInitialLengths[i];
		_bondTotalForces[i] = vNormalForce + vTangentialForce;

		CVector3 vUnsymMoment = rAC * vTangentialForce;
		_bondPrevBonds[i] = currentBond;

		// apply forces and moments directly to particles, only if bond is not broken
		if (_bondActivities[i] != 0.0)
		{
			const CVector3 partForce = vNormalForce + vTangentialForce;
			const CVector3 partMoment1 = _bondNormalMoments[i] + _bondTangentialMoments[i] - vUnsymMoment;
			const CVector3 partMoment2 = _bondNormalMoments[i] + _bondTangentialMoments[i] + vUnsymMoment;
			CUDA_VECTOR3_ATOMIC_ADD(_partForces[_bondLeftIDs[i]], partForce);
			CUDA_VECTOR3_ATOMIC_ADD(_partMoments[_bondLeftIDs[i]], partMoment1);
			CUDA_VECTOR3_ATOMIC_SUB(_partForces[_bondRightIDs[i]], partForce);
			CUDA_VECTOR3_ATOMIC_SUB(_partMoments[_bondRightIDs[i]], partMoment2);
		}
	}
}