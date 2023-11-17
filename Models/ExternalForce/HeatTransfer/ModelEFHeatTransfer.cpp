/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFHeatTransfer.h"

CModelEFHeatTransfer::CModelEFHeatTransfer()
{
	m_name = "Heat Transfer";
	m_uniqueKey = "1C8F05E051634AA78BF1359006D61003";
	m_requieredVariables.bThermals = true;
	m_hasGPUSupport = true;

	/* 0 */ AddParameter("AIR_TEMPERATURE_INIT"     , "Init environment temperature [K]"       , 1000);
	/* 1 */ AddParameter("AIR_TEMPERATURE_FINAL"    , "Final environment temperature [K]"      , 1000);
	/* 2 */ AddParameter("HEATING_TIME"             , "Time for heating from init to final [s]", 1   );
	/* 3 */ AddParameter("HEAT_TRANSFER_COEFFICIENT", "Heat transfer coefficient [W/(m2*K)]"   , 5   );
	/* 4 */ AddParameter("SURFACE_EMISSIVITY"       , "Surface emissivity [-]"                 , 0.8 );
	/* 5 */ AddParameter("SCALING_FACTOR"           , "Scaling factor (mass and size) [-]"     , 1.0 );
}

void CModelEFHeatTransfer::CalculateEF(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const
{
	// TODO: allow material-specific external force models
	// HACK: heat capacity of the material is not an integer -indicator that this particle belongs to outer layer
	//if (std::round(_particles.HeatCapacity(_iPart)) == _particles.HeatCapacity(_iPart))
		//return;

	const double partTemperature = Particles().Temperature(_iPart);
	double environmentTemperature;
	if (_time > m_parameters[2].value)
		environmentTemperature = m_parameters[1].value;
	else
		environmentTemperature = m_parameters[0].value + _time * (m_parameters[1].value - m_parameters[0].value) / m_parameters[2].value;

	const double surface = PI * std::pow(Particles().Radius(_iPart), 2);
	const double heatFluxConvection = m_parameters[3].value * surface * (environmentTemperature - partTemperature);
	const double heatFluxRadiation = m_parameters[4].value * surface * (pow(environmentTemperature, 4) - pow(partTemperature, 4));
	_particles.HeatFlux(_iPart) += m_parameters[5].value * (heatFluxConvection + heatFluxRadiation);

	// A version with temperature-dependent heat capacity.
	// To omit influence of material's heat capacity, set it in materials editor to something like 1.000001 (may not be integer).
	//const double tempCelcius = partTemperature - 273.15;
	//const double heatCapacity = 1117 + 0.14 * tempCelcius - 411 * exp(-0.006 * tempCelcius);
	//_particles.HeatFlux(_iPart) += m_parameters[5].value * (heatFluxConvection + heatFluxRadiation) / heatCapacity;
}
