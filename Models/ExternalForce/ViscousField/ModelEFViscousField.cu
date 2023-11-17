/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFViscousField.cuh"
#include "ModelEFViscousField.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[2];

void CModelEFViscousField::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelEFViscousField::CalculateEFGPU(double _time, double _timeStep, SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcExtForce_VF_kernel,
		static_cast<unsigned>(_particles.nElements),

		_particles.Radii,
		_particles.Vels,

		_particles.Forces
	);
}

void __global__ CUDA_CalcExtForce_VF_kernel(
	const unsigned _partsNum,

	const double _partRadii[],
	const CVector3 _partVels[],

	CVector3 _partForces[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partsNum; iPart += blockDim.x * gridDim.x)
	{
		double dKinViscosity = m_vConstantModelParameters[0];
		double dMediumDensity = m_vConstantModelParameters[1];

		CVector3 vRelVel = _partVels[iPart];
		double dRelVelLength = vRelVel.Length();
		double dReynolds = _partRadii[iPart] * 2 * dRelVelLength / dKinViscosity;
		if (dReynolds == 0) continue; // no drag
		double dCd;
		if (dReynolds < 0.5)
			dCd = 24.0 / dReynolds;
		else if (dReynolds < 10.1)
			dCd = 27.0 / pow(dReynolds, 0.8);
		else
			dCd = 17.0 / pow(dReynolds, 0.6);

		_partForces[iPart] -= vRelVel.Normalized() * dCd * PI * pow(_partRadii[iPart], 2.0) * dMediumDensity / 2 * pow(dRelVelLength, 2.0);
	}
}