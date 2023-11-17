/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Interaction.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "MaterialsDatabase.pb.h"
PRAGMA_WARNING_POP

CInteraction::CInteraction(const std::string& _sKey1, const std::string& _sKey2) :
	m_sCompoundKey1(_sKey1), m_sCompoundKey2(_sKey2)
{
	// create default interaction properties
	unsigned intTypes[] = _INT_PROPERTIES;
	std::string intNames[] = _INT_PROPERTIES_NAMES;
	std::string intUnits[] = _INT_PROPERTIES_UNITS;
	double intDefaults[] = _INT_PROPERTIES_DEFAULTS;
	for (size_t i = 0; i < _INT_PROPERTIES_NUMBER; ++i)
	{
		m_vProperties.push_back(new CTPProperty(intTypes[i], intNames[i], intUnits[i]));
		m_vProperties.back()->AddCorrelation(intDefaults[i]);
	}
}

CInteraction::CInteraction(const CInteraction& _interaction)
{
	Copy(_interaction);
}

CInteraction::~CInteraction()
{
	Clear();
}

CInteraction& CInteraction::operator=(const CInteraction& _interaction)
{
	if (this != &_interaction)
	{
		Clear();
		Copy(_interaction);
	}
	return *this;
}

void CInteraction::Clear()
{
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		delete m_vProperties[i];
	m_vProperties.clear();
}

void CInteraction::Copy(const CInteraction& _interaction)
{
	m_sCompoundKey1 = _interaction.m_sCompoundKey1;
	m_sCompoundKey2 = _interaction.m_sCompoundKey2;
	for (auto p : _interaction.m_vProperties)
		m_vProperties.push_back(new CTPProperty(*p));
}

std::string CInteraction::GetKey1() const
{
	return m_sCompoundKey1;
}

std::string CInteraction::GetKey2() const
{
	return m_sCompoundKey2;
}

void CInteraction::SetKeys(const std::string& _sKey1, const std::string& _sKey2)
{
	m_sCompoundKey1 = _sKey1;
	m_sCompoundKey2 = _sKey2;
}

size_t CInteraction::PropertiesNumber() const
{
	return m_vProperties.size();
}

CTPProperty* CInteraction::GetProperty(unsigned _nType)
{
	return const_cast<CTPProperty*>(static_cast<const CInteraction&>(*this).GetProperty(_nType));
}

const CTPProperty* CInteraction::GetProperty(unsigned _nType) const
{
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		if (m_vProperties[i]->GetType() == _nType)
			return m_vProperties[i];
	return nullptr;
}

CTPProperty* CInteraction::GetPropertyByIndex(size_t _nIndex)
{
	return const_cast<CTPProperty*>(static_cast<const CInteraction&>(*this).GetPropertyByIndex(_nIndex));
}

const CTPProperty* CInteraction::GetPropertyByIndex(size_t _nIndex) const
{
	if (_nIndex < m_vProperties.size())
		return m_vProperties[_nIndex];
	return nullptr;
}

double CInteraction::GetTPPropertyValue(unsigned _nType, double _dT, double _dP) const
{
	if (const CTPProperty *prop = GetProperty(_nType))
		return prop->GetValue(_dT, _dP);
	return _NOT_A_NUMBER;
}

double CInteraction::GetPropertyValue(unsigned _nType) const
{
	if (const CTPProperty *prop = GetProperty(_nType))
		return prop->GetValue();
	return _NOT_A_NUMBER;
}

void CInteraction::SetTPPropertyCorrelation(unsigned _nPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams)
{
	if (CTPProperty *prop = GetProperty(_nPropType))
		prop->AddCorrelation(_dT1, _dT2, _dP1, _dP2, _nCorrType, _vParams);
}

void CInteraction::SetPropertyValue(unsigned _nPropType, double _dValue)
{
	if (CTPProperty *prop = GetProperty(_nPropType))
		prop->SetCorrelation(_dValue);
}

void CInteraction::InitializeConditions(double _dT, double _dP)
{
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		m_vProperties[i]->InitializeConditions(_dT, _dP);
}

bool CInteraction::IsBetween(const std::string& _sKey1, const std::string& _sKey2) const
{
	return (((m_sCompoundKey1 == _sKey1) && (m_sCompoundKey2 == _sKey2)) || ((m_sCompoundKey1 == _sKey2) && (m_sCompoundKey2 == _sKey1)));
}

void CInteraction::SaveToProtobuf(ProtoInteraction& _protoInteraction)
{
	_protoInteraction.set_compoundkey1(m_sCompoundKey1);
	_protoInteraction.set_compoundkey2(m_sCompoundKey2);
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		m_vProperties[i]->SaveToProtobuf(*_protoInteraction.add_property());
}

void CInteraction::LoadFromProtobuf(const ProtoInteraction& _protoInteraction)
{
	m_sCompoundKey1 = _protoInteraction.compoundkey1();
	m_sCompoundKey2 = _protoInteraction.compoundkey2();
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		for (int j = 0; j < _protoInteraction.property_size(); ++j)
			if (m_vProperties[i]->GetType() == _protoInteraction.property(j).propertytype())
			{
				m_vProperties[i]->LoadFromProtobuf(_protoInteraction.property(j));
				break;
			}
}
