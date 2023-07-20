/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFHeatTransfer.cuh"
#include "ModelEFHeatTransfer.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[6];

void CModelEFHeatTransfer::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelEFHeatTransfer::CalculateEFGPU(double _time, double _timeStep, SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcExtForce_HT_kernel,
		static_cast<unsigned>(_particles.nElements),
		_time,
		_particles.Radii,
		_particles.HeatCapacities,
		_particles.Temperatures,
		_particles.HeatFluxes
	);
}

void __global__ CUDA_CalcExtForce_HT_kernel(
	unsigned _partsNum,
	const double _time,
	const double _partRadii[],
	const double _partHeatCapacities[],
	const double _partTemperatures[],
	double _partHeatFluxes[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partsNum; iPart += blockDim.x * gridDim.x)
	{
		// TODO: allow material-specific external force models
		// HACK: heat capacity of the material is not an integer -indicator that this particle belongs to outer layer
		//if (round(_partHeatCapacities[iPart]) == _partHeatCapacities[iPart]) // indicator that this particle belongs to outer layer
			//continue;

		const double partTemperature = _partTemperatures[iPart];
		double environmentTemperature;
		if (_time > m_vConstantModelParameters[2])
			environmentTemperature = m_vConstantModelParameters[1];
		else
			environmentTemperature = m_vConstantModelParameters[0] + _time * (m_vConstantModelParameters[1] - m_vConstantModelParameters[0]) / m_vConstantModelParameters[2];

		const double surface = PI * pow(_partRadii[iPart], 2.0);
		const double heatFluxConvection = m_vConstantModelParameters[3] * surface * (environmentTemperature - partTemperature);
		const double heatFluxRadiation = m_vConstantModelParameters[4] * surface * (pow(environmentTemperature, 4.0) - pow(partTemperature, 4.0));

		_partHeatFluxes[iPart] += m_vConstantModelParameters[5] * (heatFluxConvection + heatFluxRadiation);

		// A version with temperature-dependent heat capacity.
		// To omit influence of material's heat capacity, set it in materials editor to something like 1.000001 (may not be integer).
		//const double tempCelcius = partTemperature - 273.15;
		//const double heatCapacity = 1117 + 0.14 * tempCelcius - 411 * exp(-0.006 * tempCelcius);
		//_partHeatFluxes[iPart] += m_vConstantModelParameters[5] * (heatFluxConvection + heatFluxRadiation) / heatCapacity;
	}
}