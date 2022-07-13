#include "ModelPPHeatConduction.cuh"
#include "ModelPPHeatConduction.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[4];

void CModelPPHeatConduction::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPHeatConduction::CalculatePPHeatTransferGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPHeatTransfer_HC_kernel,
		_particles.Temperatures,

		_particles.HeatFluxes,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps
	);
}

void __global__ CUDA_CalcPPHeatTransfer_HC_kernel(
	const double	_partTemperatures[],

	double	_partHeatFluxes[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		// constants
		const unsigned iColl = _collActivityIndices[iActivColl];

		const double srcTemperature = _partTemperatures[_collSrcIDs[iColl]];
		const double dstTemperature = _partTemperatures[_collDstIDs[iColl]];
		const double contactRadius = 2 * sqrt(4 * _collEquivRadii[iColl] * _collNormalOverlaps[iColl]);

		const double temperatureK = (dstTemperature + srcTemperature) / 2;
		const double tempCelcius = temperatureK - 273.15;
		double contactThermalConductivity = m_vConstantModelParameters[0] * (5.85 + 15360 * exp(-0.002 * tempCelcius) / (tempCelcius + 516));

		const double currentOverlap = _collNormalOverlaps[iColl] / (4 * _collEquivRadii[iColl]);
		if (currentOverlap <= m_vConstantModelParameters[2])
			contactThermalConductivity *= m_vConstantModelParameters[1];
		else if (currentOverlap < m_vConstantModelParameters[3])
			contactThermalConductivity *= m_vConstantModelParameters[1] + (1 - m_vConstantModelParameters[1]) / (m_vConstantModelParameters[3] - m_vConstantModelParameters[2]) * (currentOverlap - m_vConstantModelParameters[2]);

		const double heatFlux = contactRadius * contactThermalConductivity * (dstTemperature - srcTemperature);

		CUDA_ATOMIC_ADD(_partHeatFluxes[_collSrcIDs[iColl]], heatFlux);
		CUDA_ATOMIC_SUB(_partHeatFluxes[_collDstIDs[iColl]], heatFlux);
	}
}