#include "ModelPPHeatConduction.cuh"
#include "ModelPPHeatConduction.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[5];

void CModelPPHeatConduction::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPHeatConduction::CalculatePPGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
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
		const unsigned iColl = _collActivityIndices[iActivColl];
		const double srcTemperature = _partTemperatures[_collSrcIDs[iColl]];
		const double dstTemperature = _partTemperatures[_collDstIDs[iColl]];
		const double contactThermalConductivity = m_vConstantModelParameters[4];

		// A version with temperature-dependent thermal conductivity.
		// const double temperature = (dstTemperature + srcTemperature) / 2;
		// const double temperatureCelcius = temperature - 273.15;
		// const double contactThermalConductivity = m_vConstantModelParameters[0] * (5.85 + 15360 * exp(-0.002 * temperatureCelcius) / (temperatureCelcius + 516));

		// From https://doi.org/10.1016/j.oceram.2021.100182, Equations 6-8.
		// Equations are rewritten to use equivalent radius and optimized for performance.
		const double scaledOverlap = _collNormalOverlaps[iColl] / (4 * _collEquivRadii[iColl]);
		double effectiveResistivityFactor = 1.0;
		if (scaledOverlap <= m_vConstantModelParameters[2])
			effectiveResistivityFactor = m_vConstantModelParameters[1];
		else if (scaledOverlap < m_vConstantModelParameters[3])
			effectiveResistivityFactor = m_vConstantModelParameters[1] + (1 - m_vConstantModelParameters[1]) / (m_vConstantModelParameters[3] - m_vConstantModelParameters[2]) * (scaledOverlap - m_vConstantModelParameters[2]);

		const double contactRadius = 2 * sqrt(_collEquivRadii[iColl] * _collNormalOverlaps[iColl]);
		const double heatFlux = 2 * contactRadius * m_vConstantModelParameters[0] * effectiveResistivityFactor * contactThermalConductivity * (dstTemperature - srcTemperature);

		CUDA_ATOMIC_ADD(_partHeatFluxes[_collSrcIDs[iColl]], heatFlux);
		CUDA_ATOMIC_SUB(_partHeatFluxes[_collDstIDs[iColl]], heatFlux);
	}
}