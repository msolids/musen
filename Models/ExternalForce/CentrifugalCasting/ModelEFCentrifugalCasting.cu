/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFCentrifugalCasting.cuh"
#include "ModelEFCentrifugalCasting.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[3];

void CModelEFCentrifugalCasting::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelEFCentrifugalCasting::CalculateEFGPU(double _time, double _timeStep, SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcExtForce_CS_kernel,
		static_cast<unsigned>(_particles.nElements),

		_particles.Radii,
		_particles.Coords,
		_particles.Vels,
		_particles.Masses,

		_particles.Forces
	);
}

void __global__ CUDA_CalcExtForce_CS_kernel(
	const unsigned _partsNum,

	const double   _partRadii[],
	const CVector3 _partCoords[],
	const CVector3 _partVels[],
	const double   _partMasses[],

	CVector3 _partForces[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partsNum; iPart += blockDim.x * gridDim.x)
	{
		// read model parameters
		const double dRotVelocity     = m_vConstantModelParameters[0];
		const double dLiquidDensity   = m_vConstantModelParameters[1];
		const double dLiquidViscosity = m_vConstantModelParameters[2];

		const double dParticleVolume = PI * 4.0 / 3 * pow(_partRadii[iPart], 3.0);

		// Bouyancy force (act only in Z direction)
		const double dBouyancyCentrifugalForce = (_partMasses[iPart] - dParticleVolume * dLiquidDensity) * dRotVelocity * dRotVelocity;
		CVector3 vBouyancyCentrifugalForce{
			_partCoords[iPart].x * dBouyancyCentrifugalForce,
			0,
			_partCoords[iPart].z * dBouyancyCentrifugalForce };

		const CVector3 vFluidVelocity{
			_partCoords[iPart].z * dRotVelocity,
			0,
			-_partCoords[iPart].x * dRotVelocity };

		// Friction force (from publication of Biesheuvel)
		CVector3 vFrictionForce = (vFluidVelocity - _partVels[iPart]) * (6 * PI*dLiquidViscosity*_partRadii[iPart]);

		_partForces[iPart] += vFrictionForce  + vBouyancyCentrifugalForce;
	}
}