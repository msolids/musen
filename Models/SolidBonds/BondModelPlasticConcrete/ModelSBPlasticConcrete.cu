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

	bool		_bondActivities[],
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
	/*for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _bondsNum; i += blockDim.x * gridDim.x)
	{
		if (!_bondActivities[i]) continue;

		 relative angle velocity of contact partners
		CVector3 relAngleVel = _partAnglVel[_bondLeftIDs[i]] - _partAnglVel[_bondRightIDs[i]];

		 the bond in the global coordinate system
		CVector3 currentBond = GetSolidBond(_partCoord[_bondRightIDs[i]], _partCoord[_bondLeftIDs[i]], PBC);
		double dDistanceBetweenCenters = currentBond.Length();

		 optimized
		CVector3 sumAngleVelocity = _partAnglVel[_bondLeftIDs[i]] + _partAnglVel[_bondRightIDs[i]];
		CVector3 relativeVelocity = _partVel[_bondLeftIDs[i]] - _partVel[_bondRightIDs[i]] - sumAngleVelocity*currentBond*0.5;

		CVector3 currentContact = currentBond / dDistanceBetweenCenters;
		CVector3 tempVector = _bondPrevBonds[i] * currentBond;

		CVector3 Phi = currentContact*(DotProduct(sumAngleVelocity, currentContact)*_timeStep*0.5);

		CMatrix3 M(	1 + tempVector.z*Phi.z + tempVector.y*Phi.y,	Phi.z - tempVector.z - tempVector.y*Phi.x,		-Phi.y - tempVector.z*Phi.x + tempVector.y,
					tempVector.z - Phi.z - tempVector.x*Phi.y,		tempVector.z*Phi.z + 1 + tempVector.x*Phi.x,	-tempVector.z*Phi.y + Phi.x - tempVector.x,
					-tempVector.y - tempVector.x*Phi.z + Phi.y,		-tempVector.y*Phi.z + tempVector.x - Phi.x,		tempVector.y*Phi.y + tempVector.x*Phi.x + 1);

		 normal angle velocity
		CVector3 normalAngleVel = currentContact*DotProduct(currentContact, relAngleVel);

		 calculate the force
		double dStrainTotal = (_bondInitialLengths[i] - dDistanceBetweenCenters) / _bondInitialLengths[i];
		double dNormalStress = (dStrainTotal - _bondNormalPlasticStrains[i])*_bondNormalStiffnesses[i];
		_bondNormalPlasticStrains[i] = LinearPlasticity(dStrainTotal, _bondNormalPlasticStrains[i], &dNormalStress, _bondYieldStrengths[i], _bondNormalStiffnesses[i]);
		CVector3 vNormalForce = currentContact*(dNormalStress*_bondCrossCuts[i]);

		_bondTangentialOverlaps[i] = M*_bondTangentialOverlaps[i] - (relativeVelocity - (currentContact * DotProduct(currentContact, relativeVelocity)))*_timeStep;
		_bondTangentialPlasticStrains[i] = M*_bondTangentialPlasticStrains[i];
		CVector3 vTangStress = (_bondTangentialOverlaps[i] - _bondTangentialPlasticStrains[i])*_bondTangentialStiffnesses[i];

		_bondTangentialForces[i] = vTangStress * _bondCrossCuts[i];
		_bondNormalMoments[i] = M * _bondNormalMoments[i] - normalAngleVel * (_timeStep * 2 * _bondAxialMoments[i] * _bondTangentialStiffnesses[i]) / _bondInitialLengths[i];
		_bondTangentialMoments[i] = M * _bondTangentialMoments[i] - (relAngleVel - normalAngleVel) * (_timeStep * _bondNormalStiffnesses[i] * _bondAxialMoments[i]) / _bondInitialLengths[i];

		_bondUnsymMoments[i] =  currentBond*0.5 * _bondTangentialForces[i];
		_bondPrevBonds[i] = currentBond;
		_bondTotalForces[i] = _bondTangentialForces[i] + vNormalForce;

		if (m_vConstantModelParameters[0] == 0) continue;
		 check the bond destruction
		if (dStrainTotal> 1.1)
		{
			_bondActivities[i] = false;
			_bondEndActivities[i] = _time;
		}

		double dMaxStress = -vNormalForce.Length() / _bondCrossCuts[i] + _bondTangentialMoments[i].Length() * _bondDiameters[i] / (2 * _bondAxialMoments[i]);
		double dMaxTorque = -_bondTangentialForces[i].Length() / _bondCrossCuts[i] + _bondNormalMoments[i].Length() * _bondDiameters[i] / (4 * _bondAxialMoments[i]);

		if (((fabs(dMaxStress) >= _bondNormalStrengths[i]) && ((m_vConstantModelParameters[1] != 0) || (dStrainTotal < 0))) || (fabs(dMaxTorque) >= _bondTangentialStrengths[i]))
		{
			_bondActivities[i] = false;
			_bondEndActivities[i] = _time;
		}
	}*/
}