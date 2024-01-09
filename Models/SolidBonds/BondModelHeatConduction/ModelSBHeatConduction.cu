/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBHeatConduction.cuh"
#include "ModelSBHeatConduction.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[1];
__constant__ SPBC PBC;

void CModelSBHeatConduction::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(PBC, _pbc, sizeof(SPBC));
}

void CModelSBHeatConduction::CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcSBHeatConduction_kernel,
		_particles.Coords,
		_particles.Temperatures,
		_particles.HeatFluxes,

		static_cast<unsigned>(_bonds.nElements),
		_bonds.Activities,
		_bonds.CrossCuts,
		_bonds.LeftIDs,
		_bonds.RightIDs,

		_bonds.ThermalConductivities
	);
}

__global__ void CUDA_CalcSBHeatConduction_kernel(
	const CVector3 _partCoords[],
	const double   _partTemperature[],
	double         _partHeatFlux[],

	unsigned       _bondsNum,
	uint8_t        _bondActivities[],
	const double   _bondCrossCuts[],
	const unsigned _bondLeftIDs[],
	const unsigned _bondRightIDs[],
	const double   _bondThermalConductivity[]
)
{
	for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _bondsNum; i += blockDim.x * gridDim.x)
	{
		if (!_bondActivities[i]) continue;

		const auto lID = _bondLeftIDs[i];
		const auto rID = _bondRightIDs[i];
		const double lTemperature = _partTemperature[lID];
		const double rTemperature = _partTemperature[rID];
		const double thermalConductivity = _bondThermalConductivity[i];
		const double distanceBetweenCenters = GetSolidBond(_partCoords[rID], _partCoords[lID], PBC).Length();

		const double factor = m_vConstantModelParameters[0];

		const double heatFlux= factor * _bondCrossCuts[i] * thermalConductivity * (rTemperature - lTemperature) / distanceBetweenCenters;

		CUDA_ATOMIC_ADD(_partHeatFlux[lID], heatFlux);
		CUDA_ATOMIC_SUB(_partHeatFlux[rID], heatFlux);
	}
}