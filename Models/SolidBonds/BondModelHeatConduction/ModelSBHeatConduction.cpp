/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBHeatConduction.h"
#include "BasicTypes.h"

CModelSBHeatConduction::CModelSBHeatConduction()
{
	m_name                         = "Heat Conduction";
	m_uniqueKey                    = "3BF7750E0AC54B8E909126CE48F7D85F";
	m_requieredVariables.bThermals = true;
	m_hasGPUSupport                = true;

	/* 0 */ AddParameter("CONDUCTION_SCALING_FACTOR", "Scaling factor for conductivity [-]", 1.0);
}

void CModelSBHeatConduction::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum)   const
{
	const double lTemperature = Particles().Temperature(_iLeft);
	const double rTemperature = Particles().Temperature(_iRight);
	const double thermalConductivity = Bonds().ThermalConductivity(_iBond);
	const double distanceBetweenCenters = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC).Length();

	const double factor = m_parameters[0].value;

	_bonds.HeatFlux(_iBond) = factor * Bonds().CrossCut(_iBond) * thermalConductivity * (rTemperature - lTemperature) / distanceBetweenCenters;
}

void CModelSBHeatConduction::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
{
	if (Bonds().LeftID(_iBond) == _iPart)
	{
		_particles.HeatFlux(_iPart) += Bonds().HeatFlux(_iBond);
	}
	else if (Bonds().RightID(_iBond) == _iPart)
	{
		_particles.HeatFlux(_iPart) -= Bonds().HeatFlux(_iBond);
	}
}
