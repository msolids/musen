/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFCentrifugalCasting.h"

CModelEFCentrifugalCasting::CModelEFCentrifugalCasting()
{
	m_name         = "Centrifugal Casting";
	m_uniqueKey    = "1C8F05E0516329A78BF1359006D61003";
	m_helpFileName = "/External Force/CentrifugalCasting.pdf";

	AddParameter("ROTATION_VELOCITY", "Rotation velocity [rad/s]",  1);
	//AddParameter("DRUM_RADIUS",       "Radius of drum [m]",         1);
	AddParameter("LIQUID_DENSITY",    "Density of liquid [kg/m3]",  1000);
	AddParameter("LIQUID_VISCOSITY",  "Viscosity of liquid [Pa*s]", 1.8*1e-3);

	m_hasGPUSupport = true;
}

void CModelEFCentrifugalCasting::CalculateEFForce(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const
{
	// read model parameters
	const double dRotVelocity     = m_parameters[0].value;
	const double dLiquidDensity   = m_parameters[1].value;
	const double dLiquidViscosity = m_parameters[2].value;

	const double dParticleVolume = PI * 4.0 / 3 * std::pow(Particles().Radius(_iPart), 3);

	// Bouyancy force (act only in Z direction)
	const double dBouyancyCentrifugalForce = (Particles().Mass(_iPart) - dParticleVolume * dLiquidDensity)*dRotVelocity*dRotVelocity;
	const CVector3 vBouyancyCentrifugalForce{
		Particles().Coord(_iPart).x * dBouyancyCentrifugalForce,
		0,
		Particles().Coord(_iPart).z * dBouyancyCentrifugalForce };

	const CVector3 vFluidVelocity{
		Particles().Coord(_iPart).z * dRotVelocity,
		0,
		-Particles().Coord(_iPart).x * dRotVelocity };

	// Friction force (from publication of Biesheuvel)
	const CVector3 vFrictionForce = (vFluidVelocity - Particles().Vel(_iPart)) * (6 * PI * dLiquidViscosity * Particles().Radius(_iPart));

	_particles.Force(_iPart) += vFrictionForce + vBouyancyCentrifugalForce;
}
