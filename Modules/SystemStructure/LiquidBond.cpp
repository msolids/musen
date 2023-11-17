/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "LiquidBond.h"
#include "../SimResultsStorage/DemStorage.h"

CLiquidBond::CLiquidBond(unsigned _id, CDemStorage* _storage) : CBond(_id, _storage)
{
	m_storage->Object(m_lObjectID)->set_type(ProtoParticleInfo::kLiquidBond);
	m_dViscosity = 0.1;
	m_dSurfaceTension = 73 * 1e-3; //[N/m]
	m_dContactAngle = _CONTACT_ANGLE_DEFAULT;
}

void CLiquidBond::UpdateCompoundProperties(const CCompound* _pCompound)
{
	m_dViscosity = _pCompound->GetPropertyValue(PROPERTY_DYNAMIC_VISCOSITY);
	m_dSurfaceTension = _pCompound->GetPropertyValue(PROPERTY_SURFACE_TENSION);
	UpdatePrecalculatedValues();
}

void CLiquidBond::UpdatePrecalculatedValues()
{
}
