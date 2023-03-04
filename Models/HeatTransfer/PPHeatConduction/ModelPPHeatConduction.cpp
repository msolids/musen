#include "ModelPPHeatConduction.h"

CModelPPHeatConduction::CModelPPHeatConduction()
{
	m_name = "PP Heat Conduction";
	m_uniqueKey = "B18A46C2786D4D44B925A8A04D0D1098";
	m_helpFileName = "/Contact Models/HeatConduction.pdf";
	m_hasGPUSupport = true;

	AddParameter("SCALING_PARAMETER_CONDUCTION", "Scaling parameter conduction [-]", 1e+20);
	AddParameter("FACTOR_CONDUCTIVITY", "Scaling parameter conduction [-]", 0.2);
	AddParameter("MIN_OVERLAP", "Minimal overlap [-]", 0.03);
	AddParameter("MAX_OVERLAP", "Maximal overlap [-]", 0.05);
}

void CModelPPHeatConduction::CalculatePPHeatTransfer(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// constants
	const double srcTemperature = Particles().Temperature(_iSrc);
	const double dstTemperature = Particles().Temperature(_iDst);
	const double temperatureK = (dstTemperature + srcTemperature) / 2;
	const double tempCelcius = temperatureK - 273.15;

	double contactThermalConductivity = m_parameters[0].value * (5.85 + 15360 * exp(-0.002 * tempCelcius) / (tempCelcius + 516));
	const double currentOverlap = _collision->dNormalOverlap / (4 * _collision->dEquivRadius);
	if (currentOverlap <= m_parameters[2].value)
		contactThermalConductivity *= m_parameters[1].value;
	else if (currentOverlap < m_parameters[3].value)
		contactThermalConductivity *= (m_parameters[1].value + (1 - m_parameters[1].value) / (m_parameters[3].value - m_parameters[2].value) * (currentOverlap - m_parameters[2].value));

	const double contactRadius = 2 * sqrt(4 * _collision->dEquivRadius * _collision->dNormalOverlap);
	_collision->dHeatFlux = contactRadius * contactThermalConductivity * (dstTemperature - srcTemperature);
}
