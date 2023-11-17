/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPP.cuh"
#include "ModelPP.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;

// TODO: Set required number of parameters. It must correspond to those defined in constructor CModelPP::CModelPP() with functions AddParameter().
__constant__ double m_vConstantModelParameters[1];

void CModelPP::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

/// Invokes the GPU-version of the model.
void CModelPP::CalculatePPGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	/// The function is invoked with the maximum possible set of parameters.
	/// All unused parameters may be removed (here and and in ModelPP.cuh) for speed-up.
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_kernel,
		_time,
		_timeStep,
		_interactProps,

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

		static_cast<unsigned>(_collisions.nElements),
		_collisions.ActiveCollisionsNum,
		_collisions.ActivityFlags,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.ContactVectors,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivMasses,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps,
		_collisions.VirtualShifts,

		_collisions.TangOverlaps
	);
}

/// GPU-version of the model.
/// The function is presented with the maximum possible set of parameters.
/// All unused parameters may be removed (here and and in ModelPP.cuh) for speed-up.
void __global__ CUDA_CalcPPForce_kernel(
	double					_time,
	double					_timeStep,
	const SInteractProps	_interactProps[],

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

	CVector3	_partForces[],
	CVector3	_partMoments[],

	unsigned			_collsNum,
	const unsigned*		_collActiveCollisionsNum,
	const bool			_collActivityFlags[],
	const unsigned		_collActivityIndices[],
	const uint16_t		_collInteractPropIDs[],
	const CVector3		_collContactVectors[],
	const unsigned		_collSrcIDs[],
	const unsigned		_collDstIDs[],
	const double		_collEquivMasses[],
	const double		_collEquivRadii[],
	const double		_collNormalOverlaps[],
	const uint8_t		_collVirtShifts[],

	CVector3 _collTangOverlaps[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		// TODO: Write your model here.
	}
}
