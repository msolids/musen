/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SimplifiedSceneGPU.cuh"
#include "CUDAKernels.cuh"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
PRAGMA_WARNING_POP

void CGPUScene::SetCudaDefines(const CCUDADefines* _cudaDefines)
{
	m_cudaDefines = _cudaDefines;
}

void CGPUScene::GetMaxSquaredPartVerletDistance(SGPUParticles& _particles, double* _bufMaxVel)
{
	if (!_particles.nElements)
	{
		CUDA_MEMSET(_bufMaxVel, 0, sizeof(double));
		return;
	}

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GatherSquaredPartVerletDistances_kernel, static_cast<unsigned>(_particles.nElements),
		_particles.Coords, _particles.CoordsVerlet, _particles.Activities, _particles.TempDouble1);

	CUDA_REDUCE_CALLER(CUDAKernels::ReduceMax_kernel, _particles.nElements, _particles.TempDouble1, _particles.TempDouble2, _bufMaxVel);
}

double CGPUScene::GetMaxPartVelocity(SGPUParticles& _particles) const
{
	if (!_particles.nElements)
		return 0.0;

	static thrust::device_vector<double> maxSqrVelocity;
	if (maxSqrVelocity.empty())
		maxSqrVelocity.resize(1);
	static thrust::device_vector<double> temp;
	if (temp.size() != _particles.nElements)
		temp.resize(_particles.nElements);

	CUDA_REDUCE_CALLER(CUDAKernels::ReduceMax_kernel, _particles.nElements, _particles.Vels, temp.data().get(), maxSqrVelocity.data().get());

	double maxVelocity;
	CUDA_MEMCPY_D2H(&maxVelocity, maxSqrVelocity.data().get(), sizeof(double));
	return std::sqrt(maxVelocity);
}

double CGPUScene::GetMaxPartTemperature(SGPUParticles& _particles) const
{
	if (!_particles.nElements)
		return 0.0;

	static thrust::device_vector<double> maxTemperature;
	if (maxTemperature.empty())
		maxTemperature.resize(1);
	static thrust::device_vector<double> temp;
	if (temp.size() != _particles.nElements)
		temp.resize(_particles.nElements);

	CUDA_REDUCE_CALLER(CUDAKernels::ReduceMax_kernel, _particles.nElements, _particles.Temperatures, temp.data().get(), maxTemperature.data().get());

	double res;
	CUDA_MEMCPY_D2H(&res, maxTemperature.data().get(), sizeof(double));
	return res;
}

void CGPUScene::GetMaxWallVelocity(SGPUWalls& _walls, double* _bufMaxVel) const
{
	if (!_walls.nElements)
	{
		CUDA_MEMSET(_bufMaxVel, 0, sizeof(double));
		return;
	}

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GatherWallVelocities_kernel, static_cast<unsigned>(_walls.nElements), _walls.Vertices1, _walls.Vertices2, _walls.Vertices3, _walls.Vels, _walls.RotCenters, _walls.RotVels, _walls.TempVels1);

	CUDA_REDUCE_CALLER(CUDAKernels::ReduceMax_kernel, _walls.nElements, _walls.TempVels1, _walls.TempVels2, _bufMaxVel);
}

size_t CGPUScene::GetBrokenBondsNumber(const SGPUSolidBonds& _bonds) const
{
	const thrust::device_ptr<uint8_t> activity = thrust::device_pointer_cast(_bonds.Activities);
	return thrust::count(activity, activity + _bonds.nElements, false);
}
