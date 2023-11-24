/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHeatTransfer.h"

CModelPWHeatTransfer::CModelPWHeatTransfer()
{
	m_name                         = "Heat Transfer";
	m_uniqueKey                    = "E51F4196FE24495191EA9A1A8E794925";
	m_helpFileName                 = "";
	m_requieredVariables.bThermals = true;
	m_hasGPUSupport                = true;

	/* 0 */ AddParameter("WALL_TEMPERATURE"   , "Constant temperature of the wall [K]", 1173);
	/* 1 */ AddParameter("HEAT_TRANSFER_COEFF", "Heat transfer coefficient [W/m2/K]"  , 50  );
	/* 2 */ AddParameter("RESISTIVITY_FACTOR" , "Resistivity factor [-]"              , 0.2 );
}

void CModelPWHeatTransfer::CalculatePW(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _collision) const
{
	const double partRadius      = Particles().Radius(_iPart);
	const double partTemperature = Particles().Temperature(_iPart);

	const double wallTemperature   = m_parameters[0].value;
	const double heatTransferCoeff = m_parameters[1].value;
	const double resistivityFactor = m_parameters[2].value;

	const CVector3 rc = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _collision->vContactVector;
	const double   rcLen = rc.Length();

	// normal overlap
	const double normOverlap = partRadius - rcLen;
	if (normOverlap < 0) return;

	_collision->dHeatFlux = PI * partRadius * normOverlap * heatTransferCoeff * resistivityFactor * (wallTemperature - partTemperature);
}

void CModelPWHeatTransfer::ConsolidatePart(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.HeatFlux(_iPart) += _collision->dHeatFlux;
}
