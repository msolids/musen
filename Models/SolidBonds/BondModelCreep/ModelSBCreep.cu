/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBCreep.cuh"
#include "ModelSBCreep.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[7];
__constant__ SPBC PBC;


void CModelSBCreep::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelSBCreep::CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcSBForce_CRP_kernel,
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



__global__ void CUDA_CalcSBForce_CRP_kernel(
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


		double dCreep_A = m_vConstantModelParameters[4];
		double dCreep_m = m_vConstantModelParameters[5];
		double dFractureByStress = m_vConstantModelParameters[6];

		// relative angle velocity of contact partners
		CVector3 relAngleVel = _partAnglVels[_bondLeftIDs[i]] - _partAnglVels[_bondRightIDs[i]];

		// the bond in the global coordinate system
		CVector3 currentBond = GetSolidBond(_partCoords[_bondRightIDs[i]], _partCoords[_bondLeftIDs[i]], PBC);
		double dDistanceBetweenCenters = currentBond.Length();
		double dBondInitLength = _bondInitialLengths[i];
		const double dBondCrossCut = _bondCrossCuts[i];
		const double dBondAxialMoment = _bondAxialMoments[i];

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
		double dStrainTotal = (dDistanceBetweenCenters - dBondInitLength) / dBondInitLength;
		double dNormalStress = (dStrainTotal - _bondNormalPlasticStrains[i])*_bondNormalStiffnesses[i];

		double dYieldStrength = _bondYieldStrengths[i];
		if (dNormalStress < 0) // compression
			dYieldStrength *= m_vConstantModelParameters[3];
		if (dNormalStress > dYieldStrength)
		{
			_bondNormalPlasticStrains[i] = dStrainTotal - dYieldStrength / _bondNormalStiffnesses[i];
			dNormalStress = dYieldStrength;
		}
		else if (dNormalStress < -dYieldStrength)
		{
			_bondNormalPlasticStrains[i] = dStrainTotal + dYieldStrength / _bondNormalStiffnesses[i];
			dNormalStress = -dYieldStrength;
		}
		CVector3 vNormalForce = currentContact * (dNormalStress*-1 * dBondCrossCut);

		// creep in normal direction
		if (dNormalStress > 0)
			_bondNormalPlasticStrains[i] += _timeStep *  dCreep_A * pow(fabs(dNormalStress), dCreep_m);
		else
			_bondNormalPlasticStrains[i] -= _timeStep *  dCreep_A * pow(fabs(dNormalStress), dCreep_m);


		_bondTangentialOverlaps[i] = M * _bondTangentialOverlaps[i] - tangentialVelocity * _timeStep;
		double dTangentialStress = Length(_bondTangentialOverlaps[i] * _bondTangentialStiffnesses[i] / dBondInitLength);

		if( m_vConstantModelParameters[2] ) // tangential yield
			if (dTangentialStress > dYieldStrength)
			{
				_bondTangentialOverlaps[i] *= dYieldStrength * dBondInitLength / (_bondTangentialStiffnesses[i] * Length(_bondTangentialOverlaps[i]));
				dTangentialStress = dYieldStrength;
			}

		// tangential creep
		if (_bondTangentialOverlaps[i].SquaredLength() > 0)
			_bondTangentialOverlaps[i] *= (1 - _timeStep * dCreep_A * pow(fabs(dTangentialStress), dCreep_m) * dBondInitLength/ _bondTangentialOverlaps[i].Length());

		const CVector3 vTangentialForce = _bondTangentialOverlaps[i] * (_bondTangentialStiffnesses[i] * dBondCrossCut / dBondInitLength);
		const CVector3 vBondNormalMoment = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * dBondAxialMoment * _bondTangentialStiffnesses[i] / dBondInitLength);
		const CVector3 vBondTangentialMoment = M * _bondTangentialMoments[i] - tangAngleVel * (_timeStep * dBondAxialMoment * _bondNormalStiffnesses[i] / dBondInitLength);

		_bondNormalMoments[i] = vBondNormalMoment;
		_bondTangentialMoments[i] = vBondTangentialMoment;

		const CVector3 vUnsymMoment = currentBond * 0.5 * vTangentialForce;
		_bondPrevBonds[i] = currentBond;
		_bondTotalForces[i] = vNormalForce + vTangentialForce;

		bool bondBreaks = false;

		// check fracture condition caused by strain
		if ((dStrainTotal > m_vConstantModelParameters[0])
			|| (m_vConstantModelParameters[1] && (dStrainTotal < -m_vConstantModelParameters[1]))) // strain greater than breakage strain
			bondBreaks = true;

		if (dFractureByStress) // check fracture condition caused by stress
		{
			// check the bond destruction
			double forceLength = vNormalForce.Length();
			if (dStrainTotal <= 0)	// compression
				forceLength *= -1;
			const double maxStress1 = forceLength / dBondCrossCut;
			const double maxStress2 = vBondTangentialMoment.Length() * _bondDiameters[i] / (2.0 * dBondAxialMoment);
			const double maxTorque1 = vTangentialForce.Length() / dBondCrossCut;
			const double maxTorque2 = vBondNormalMoment.Length() * _bondDiameters[i] / (4.0 * dBondAxialMoment);

			if ((maxStress1 + maxStress2 >= _bondNormalStrengths[i]) || (maxTorque1 + maxTorque2 >= _bondTangentialStrengths[i]))
				bondBreaks = true;
		}

		if (bondBreaks)
		{
			_bondActivities[i] = false;
			_bondEndActivities[i] = _time;
			continue; // if bond is broken do not apply forces and moments
		}

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