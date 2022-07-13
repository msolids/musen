#include "ModelEFHeatTransfer.cuh"
#include "ModelEFHeatTransfer.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[6];

void CModelEFHeatTransfer::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelEFHeatTransfer::CalculateEFForceGPU(double _time, double _timeStep, SGPUParticles& _particles)
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
		if (round(_partHeatCapacities[iPart]) != _partHeatCapacities[iPart]) // indicator that this particle belongs to outer layer
		{
			double environmentTemperature;
			if (_time > m_vConstantModelParameters[2])
				environmentTemperature = m_vConstantModelParameters[1];
			else
				environmentTemperature = m_vConstantModelParameters[0] + _time * (m_vConstantModelParameters[1] - m_vConstantModelParameters[0]) / m_vConstantModelParameters[2];

			const double surface = PI * pow(_partRadii[iPart], 2.0);
			const double heatFluxConvection = surface * m_vConstantModelParameters[3] * (environmentTemperature - _partTemperatures[iPart]);
			const double heatFluxRadiation = 5.67 * 1e-5 * m_vConstantModelParameters[4] * surface * (pow(environmentTemperature, 4.0) - pow(_partTemperatures[iPart], 4.0));

			_partHeatFluxes[iPart] += m_vConstantModelParameters[5] * (heatFluxConvection + heatFluxRadiation);
		}
	}
}