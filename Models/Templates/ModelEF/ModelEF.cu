/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEF.cuh"
#include "ModelEF.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;

// TODO: Set required number of parameters. It must correspond to those defined in constructor CModelPP::CModelPP() with functions AddParameter().
__constant__ double m_vConstantModelParameters[1];

void CModelEF::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

/// Invokes the GPU-version of the model.
void CModelEF::CalculateEFGPU(double _time, double _timeStep, SGPUParticles& _particles)
{
	/// The function is invoked with the maximum possible set of parameters.
	/// All unused parameters may be removed (here and and in ModelEF.cuh) for speed-up.
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcEFForce_kernel,
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
		_particles.Moments,
		_particles.Quaternions,
		_particles.Radii,
		_particles.Vels,

		_particles.Forces
	);
}

/// GPU-version of the model.
/// The function is presented with the maximum possible set of parameters.
/// All unused parameters may be removed (here and and in ModelEF.cuh) for speed-up.
void __global__ CUDA_CalcEFForce_kernel(
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
	const CVector3		_partMoments[],
	const CQuaternion	_partQuaternions[],
	const double		_partRadii[],
	const CVector3		_partVels[],

	CVector3 _partForces[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partsNum; iPart += blockDim.x * gridDim.x)
	{
		// TODO: Write your model here.
	}
}
