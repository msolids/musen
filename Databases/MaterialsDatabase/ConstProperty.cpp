/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ConstProperty.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "GeneratedFiles/MaterialsDatabase.pb.h"
PRAGMA_WARNING_POP

CConstProperty::CConstProperty(void)
	: CBaseProperty(),
	m_dValue(0)
{
}

CConstProperty::CConstProperty(unsigned _nType, const std::string& _sName, const std::string& _sUnits)
	: CBaseProperty(_nType, _sName, _sUnits),
	m_dValue(0)
{
}

CConstProperty::~CConstProperty()
{
}

void CConstProperty::SetValue(double _dValue)
{
	m_dValue = _dValue;
}

double CConstProperty::GetValue() const
{
	return m_dValue;
}

double CConstProperty::GetValue(double _dT, double _dP) const
{
	return m_dValue;
}

void CConstProperty::SaveToProtobuf(ProtoConstProperty& _protoConstProperty)
{
	_protoConstProperty.set_propertytype(m_nPropertyType);
	_protoConstProperty.set_value(m_dValue);
}

void CConstProperty::LoadFromProtobuf(const ProtoConstProperty& _protoConstProperty)
{
	m_dValue = _protoConstProperty.value();
}