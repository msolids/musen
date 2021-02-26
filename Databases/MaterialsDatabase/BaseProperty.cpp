/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BaseProperty.h"
#include "DefinesMDB.h"

CBaseProperty::CBaseProperty():
	m_nPropertyType(PROPERTY_UNDEFINED_TYPE),
	m_sName(PROPERTY_UNDEFINED_NAME),
	m_sUnits(PROPERTY_UNDEFINED_UNITS)
{
}

CBaseProperty::CBaseProperty(unsigned _nType, const std::string& _sName, const std::string& _sUnits) :
	m_nPropertyType(_nType),
	m_sName(_sName),
	m_sUnits(_sUnits)
{
}

CBaseProperty::~CBaseProperty()
{
}

void CBaseProperty::SetUnits(const std::string& _sUnits)
{
	m_sUnits = _sUnits;
}

std::string CBaseProperty::GetUnits() const
{
	return m_sUnits;
}

void CBaseProperty::SetName(const std::string& _sName)
{
	m_sName = _sName;
}

std::string CBaseProperty::GetName() const
{
	return m_sName;
}

void CBaseProperty::SetType(unsigned _nType)
{
	m_nPropertyType = _nType;
}

unsigned CBaseProperty::GetType() const
{
	return m_nPropertyType;
}

double CBaseProperty::GetValue() const
{
	return _NOT_A_NUMBER;
}

double CBaseProperty::GetValue(double _dT, double _dP) const
{
	return _NOT_A_NUMBER;
}

