#include "ModelPPHeatConduction.h"

CModelPPHeatConduction::CModelPPHeatConduction()
{
	m_name = "PP Heat Conduction";
	m_uniqueKey = "B18A46C2786D4D44B925A8A04D0D1098";
	m_helpFileName = "/Contact Models/HeatConduction.pdf";
	m_requieredVariables.bThermals = true;
	m_hasGPUSupport = true;

	/* 0 */ AddParameter("CONDUCTION_SCALING_FACTOR", "Scaling factor for conductivity [-]"      , 1.0 );
	/* 1 */ AddParameter("RESISTIVITY_FACTOR"       , "Resistivity factor [-]"                   , 0.2 );
	/* 2 */ AddParameter("MIN_OVERLAP"              , "Minimum overlap factor [-]"               , 0.03);
	/* 3 */ AddParameter("MAX_OVERLAP"              , "Maximum overlap factor [-]"               , 0.05);
	/* 4 */ AddParameter("THERMAL_CONDUCTIVITY"     , "Thermal conductivity in contact [W/(m*K)]", 25  );
}

void CModelPPHeatConduction::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	const double srcTemperature = Particles().Temperature(_iSrc);
	const double dstTemperature = Particles().Temperature(_iDst);
	const double contactThermalConductivity = m_parameters[4].value;

	// A version with temperature-dependent thermal conductivity.
	// const double temperature = (dstTemperature + srcTemperature) / 2;
	// const double temperatureCelcius = temperature - 273.15;
	// const double contactThermalConductivity = m_parameters[0].value * (5.85 + 15360 * exp(-0.002 * temperatureCelcius) / (temperatureCelcius + 516));

	// From https://doi.org/10.1016/j.oceram.2021.100182, Equations 6-8.
	// Equations are rewritten to use equivalent radius and optimized for performance.
	const double scaledOverlap = _collision->dNormalOverlap / (4 * _collision->dEquivRadius);
	double effectiveResistivityFactor = 1.0;
	if (scaledOverlap <= m_parameters[2].value)
		effectiveResistivityFactor = m_parameters[1].value;
	else if (scaledOverlap < m_parameters[3].value)
		effectiveResistivityFactor = m_parameters[1].value + (1 - m_parameters[1].value) / (m_parameters[3].value - m_parameters[2].value) * (scaledOverlap - m_parameters[2].value);

	const double contactRadius = 2 * sqrt(_collision->dEquivRadius * _collision->dNormalOverlap);
	_collision->dHeatFlux = 2 * contactRadius * m_parameters[0].value * effectiveResistivityFactor * contactThermalConductivity * (dstTemperature - srcTemperature);
}

void CModelPPHeatConduction::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.HeatFlux(_iPart) += _collision->dHeatFlux;
}

void CModelPPHeatConduction::ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.HeatFlux(_iPart) -= _collision->dHeatFlux;
}
