/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SolidBond.h"
#include "../SimResultsStorage/DemStorage.h"

CSolidBond::CSolidBond(unsigned _id, CDemStorage* _storage) : CBond(_id, _storage)
{
	m_storage->Object(m_lObjectID)->set_type(ProtoParticleInfo::kBond);

	m_dYoungModulus = _YOUNG_MODULUS_DEFAULT;
	m_dPoissonRatio = _POISSON_RATIO_DEFAULT;
	m_dViscosity = 1e+4;
	m_dNormalStrength = 1e+11;
	m_dTangentialStrength = 1e+11;
	m_dShearModulus = _YOUNG_MODULUS_DEFAULT / (2 * (1 + _POISSON_RATIO_DEFAULT));
	m_dTimeThermExpCoeff = _TIME_THERM_EXP_DEFAULT;
	m_dYieldStrength = _YIELD_DEFAULT;
	m_thermalConductivity = _THERMAL_CONDUCTIVITY_DEFAULT;
	UpdatePrecalculatedValues();
}

void CSolidBond::CloneData(const CPhysicalObject& _other)
{
	CBond::CloneData(_other);
	if (_other.GetObjectType() != SOLID_BOND)
		return;
	const auto& other = dynamic_cast<const CSolidBond&>(_other);
	m_dYoungModulus       = other.m_dYoungModulus;
	m_dPoissonRatio       = other.m_dPoissonRatio;
	m_dShearModulus       = other.m_dShearModulus;
	m_dNormalStrength     = other.m_dNormalStrength;
	m_dTangentialStrength = other.m_dTangentialStrength;
	m_dTimeThermExpCoeff  = other.m_dTimeThermExpCoeff;
	m_dYieldStrength      = other.m_dYieldStrength;
	m_thermalConductivity = other.m_thermalConductivity;
	m_dAxialMoment        = other.m_dAxialMoment;
	m_dCrossCutSurface    = other.m_dCrossCutSurface;
	m_NormalForce         = other.m_NormalForce;
	m_TangentialForce     = other.m_TangentialForce;
	m_NormalMoment        = other.m_NormalMoment;
	m_TangentialMoment    = other.m_TangentialMoment;
	m_TangDisplacement    = other.m_TangDisplacement;
}

void CSolidBond::UpdateCompoundProperties(const CCompound* _pCompound)
{
	m_dYoungModulus = _pCompound->GetPropertyValue(PROPERTY_YOUNG_MODULUS);
	m_dPoissonRatio = _pCompound->GetPropertyValue(PROPERTY_POISSON_RATIO);
	m_dTangentialStrength = _pCompound->GetPropertyValue(PROPERTY_TANGENTIAL_STRENGTH);
	m_dNormalStrength = _pCompound->GetPropertyValue(PROPERTY_NORMAL_STRENGTH);
	m_dViscosity = _pCompound->GetPropertyValue(PROPERTY_DYNAMIC_VISCOSITY);
	m_dTimeThermExpCoeff = _pCompound->GetPropertyValue( PROPERTY_TIME_THERM_EXP_COEFF );
	m_dYieldStrength = _pCompound->GetPropertyValue( PROPERTY_YIELD_STRENGTH );
	m_thermalConductivity = _pCompound->GetPropertyValue(PROPERTY_THERMAL_CONDUCTIVITY);
	m_dShearModulus = m_dYoungModulus / (2 * (1 + m_dPoissonRatio));
	UpdatePrecalculatedValues();
}

void CSolidBond::UpdatePrecalculatedValues()
{
	m_dCrossCutSurface = PI * m_dDiameter * m_dDiameter / 4;
	m_dAxialMoment = PI * m_dDiameter * m_dDiameter * m_dDiameter * m_dDiameter / 64; // I
}