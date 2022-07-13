#include "ModelEFHeatTransfer.h"

CModelEFHeatTransfer::CModelEFHeatTransfer()
{
	m_name = "Heat Transfer";
	m_uniqueKey = "1C8F05E051634AA78BF1359006D61003";
	m_requieredVariables.bThermals = true;
	m_hasGPUSupport = true;

	AddParameter("AIR_TEMPERATURE_INIT", "Init environment temperature [K]", 1000);
	AddParameter("AIR_TEMPERATURE_FINAL", "Final environment temperature [K]", 1000);
	AddParameter("HEATING_TIME", "Time for heating from init to final [s]", 1000);
	AddParameter("HEAT_TRANSFER_COEFFICIENT", "Heat transfer coefficient [W/(m2*K)]", 5);
	AddParameter("SURFACE_EMISSIVITY", "Surface emissivity of furnace [-]", 0.8);
	AddParameter("SCALING_FACTOR", "Scaling factor (mass and size) [-]", 0.5e+18);
}

void CModelEFHeatTransfer::CalculateEFForce(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const
{
	if (std::round(_particles.HeatCapacity(_iPart)) != _particles.HeatCapacity(_iPart)) // indicator that this particle belongs to outer layer
	{
		double environmentTemperature;
		if (_time > m_parameters[2].value)
			environmentTemperature = m_parameters[1].value;
		else
			environmentTemperature = m_parameters[0].value + _time * (m_parameters[1].value - m_parameters[0].value) / m_parameters[2].value;
		const double surface = PI * std::pow(Particles().Radius(_iPart), 2);
		const double heatFluxConvection = surface * m_parameters[3].value * (environmentTemperature - Particles().Temperature(_iPart));
		const double heatFluxRadiation = 5.67 * 1e-5 * m_parameters[4].value * surface * (pow(environmentTemperature, 4) - pow(Particles().Temperature(_iPart), 4));
		const double tempCelcius = Particles().Temperature(_iPart) - 273.15;
		const double heatCapacity = 1117 + 0.14 * tempCelcius - 411 * exp(-0.006 * tempCelcius);
		_particles.Temperature(_iPart) += m_parameters[5].value * (heatFluxConvection + heatFluxRadiation) * _timeStep / (heatCapacity * _particles.Mass(_iPart));
	}
}
