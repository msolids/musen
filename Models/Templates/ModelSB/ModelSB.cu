/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSB.cuh"
#include "ModelSB.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;

// TODO: Set required number of parameters. It must correspond to those defined in constructor CModelPP::CModelPP() with functions AddParameter().
__constant__ double m_vConstantModelParameters[1];

void CModelSB::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

/// Invokes the GPU-version of the model.
void CModelSB::CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	/// The function is invoked with the maximum possible set of parameters.
	/// All unused parameters may be removed (here and and in ModelSB.cuh) for speed-up.
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcSBForce_kernel,
		_time,
		_timeStep,

		static_cast<unsigned>(_particles.nElements),
		_particles.Activities,
		_particles.AnglVels,
		_particles.CompoundIndices,
		_particles.ContactRadii,
		_particles.Coords,
		_particles.EndActivities,
		_particles.InertiaMoments,
		_particles.Masses,
		_particles.Quaternions,
		_particles.Radii,
		_particles.Vels,
		_particles.Forces,
		_particles.Moments,

		static_cast<unsigned>(_bonds.nElements),
		_bonds.AxialMoments,
		_bonds.CrossCuts,
		_bonds.Diameters,
		_bonds.InitialIndices,
		_bonds.InitialLengths,
		_bonds.LeftIDs,
		_bonds.RightIDs,
		_bonds.NormalStiffnesses,
		_bonds.NormalStrengths,
		_bonds.TangentialStiffnesses,
		_bonds.TangentialStrengths,
		_bonds.TimeThermExpCoeffs,
		_bonds.Viscosities,
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

/// GPU-version of the model.
/// The function is presented with the maximum possible set of parameters.
/// All unused parameters may be removed (here and and in ModelSB.cuh) for speed-up.
void __global__ CUDA_CalcSBForce_kernel(
	double		_time,
	double		_timeStep,

	unsigned			_partsNum,
	const unsigned		_partActivities[],
	const CVector3		_partAnglVels[],
	const unsigned		_partCompoundIndices[],
	const double		_partContactRadii[],
	const CVector3		_partCoords[],
	const double		_partEndActivities[],
	const double		_partInertiaMoments[],
	const double		_partMasses[],
	const CQuaternion	_partQuaternions[],
	const double		_partRadii[],
	const CVector3		_partVels[],
	CVector3			_partForces[],
	CVector3			_partMoments[],

	unsigned			_bondsNum,
	const double		_bondAxialMoments[],
	const double		_bondCrossCuts[],
	const double		_bondDiameters[],
	const unsigned		_bondInitialIndices[],
	const double		_bondInitialLengths[],
	const unsigned		_bondLeftIDs[],
	const unsigned		_bondRightIDs[],
	const double		_bondNormalStiffnesses[],
	const double		_bondNormalStrengths[],
	const double		_bondTangentialStiffnesses[],
	const double		_bondTangentialStrengths[],
	const double		_bondTimeThermExpCoeffs[],
	const double		_bondViscosities[],
	const double		_bondYieldStrengths[],

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
	for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _bondsNum; i += blockDim.x * gridDim.x)
	{
		if (!_bondActivities[i]) continue;

		// TODO: Write your model here.
	}
}
