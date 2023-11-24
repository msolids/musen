/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPW.cuh"
#include "ModelPW.h"
#include <device_launch_parameters.h>

__constant__ SPBC PBC;

// TODO: Set required number of parameters. It must correspond to those defined in constructor CModelPW::CModelPW() with functions AddParameter().
__constant__ double m_vConstantModelParameters[1];

void CModelPW::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

/// Invokes the GPU-version of the model.
void CModelPW::CalculatePWGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions)
{
	/// The function is invoked with the maximum possible set of parameters.
	/// All unused parameters may be removed (here and and in ModelPW.cuh) for speed-up.
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPWForce_kernel,
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
		_particles.Forces,
		_particles.InertiaMoments,
		_particles.Masses,
		_particles.Quaternions,
		_particles.Radii,
		_particles.Vels,

		_particles.Moments,

		static_cast<unsigned>(_walls.nElements),
		_walls.CompoundIndices,
		_walls.Forces,
		_walls.MaxCoords,
		_walls.MinCoords,
		_walls.NormalVectors,
		_walls.RotCenters,
		_walls.RotVels,
		_walls.Vels,
		_walls.Vertices1,
		_walls.Vertices2,
		_walls.Vertices3,

		static_cast<unsigned>(_collisions.nElements),
		_collisions.ActiveCollisionsNum,
		_collisions.ActivityFlags,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.ContactVectors,  // interpreted as Contact Point
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivMasses,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps,
		_collisions.VirtualShifts,

		_collisions.TangOverlaps,
		_collisions.TotalForces
	);
}

/// GPU-version of the model.
/// The function is presented with the maximum possible set of parameters.
/// All unused parameters may be removed (here and and in ModelPW.cuh) for speed-up.
void __global__ CUDA_CalcPWForce_kernel(
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
	const CVector3		_partForces[],
	const double		_partInertiaMoments[],
	const double		_partMasses[],
	const CQuaternion	_partQuaternions[],
	const double		_partRadii[],
	const CVector3		_partVels[],

	CVector3			_partMoments[],

	unsigned			_wallsNum,
	const unsigned		_wallCompoundIndices[],
	const CVector3		_wallForces[],
	const CVector3		_wallMaxCoords[],
	const CVector3		_wallMinCoords[],
	const CVector3		_wallNormalVectors[],
	const CVector3		_wallRotCenters[],
	const CVector3		_wallRotVels[],
	const CVector3		_wallVels[],
	const CVector3		_wallVertices1[],
	const CVector3		_wallVertices2[],
	const CVector3		_wallVertices3[],

	unsigned			_collsNum,
	const unsigned*		_collActiveCollisionsNum,
	const bool			_collActivityFlags[],
	const unsigned		_collActivityIndices[],
	const uint16_t		_collInteractPropIDs[],
	const CVector3		_collContactPoints[],
	const unsigned		_collSrcIDs[],
	const unsigned		_collDstIDs[],
	const double		_collEquivMasses[],
	const double		_collEquivRadii[],
	const double		_collNormalOverlaps[],
	const uint8_t		_collVirtShifts[],

	CVector3 _collTangOverlaps[],
	CVector3 _collTotalForces[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		// TODO: Write your model here.
	}
}
