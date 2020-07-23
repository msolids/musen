/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TPProperty.h"
#include "MixedFunctions.h"

CTPProperty::CTPProperty()
{
	ResetConditions();
}

CTPProperty::CTPProperty(unsigned _nType, const std::string& _sName, const std::string& _sUnits) : CBaseProperty(_nType, _sName, _sUnits)
{
	ResetConditions();
}

CTPProperty::CTPProperty(const CTPProperty& _property) : CBaseProperty(_property)
{
	Copy(_property);
	ResetConditions();
}

CTPProperty::~CTPProperty()
{
	Clear();
}

CTPProperty& CTPProperty::operator=(const CTPProperty& _property)
{
	if (this != &_property)
	{
		Clear();
		Copy(_property);
		ResetConditions();
	}
	return *this;
}

void CTPProperty::Clear()
{
	m_vCorrelations.clear();
}

void CTPProperty::Copy(const CTPProperty& _property)
{
	for (auto const& c : _property.m_vCorrelations)
		m_vCorrelations.push_back(std::make_unique<CCorrelation>(*c));
}

double CTPProperty::GetValue(double _dT, double _dP) const
{
	if ((m_nLastUsedCorrelation != -1) && (m_vCorrelations[m_nLastUsedCorrelation]->IsInInterval(_dT, _dP)))
		return m_dLastReturnedValue = m_vCorrelations[m_nLastUsedCorrelation]->GetCorrelationValue(_dT, _dP);

	for (size_t i = 0; i < m_vCorrelations.size(); ++i)
		if (m_vCorrelations[i]->IsInInterval(_dT, _dP))
		{
			m_nLastUsedCorrelation = static_cast<int>(i);
			return m_dLastReturnedValue = m_vCorrelations[i]->GetCorrelationValue(_dT, _dP);
		}
	return m_dLastReturnedValue = _NOT_A_NUMBER;
}

double CTPProperty::GetValue() const
{
	if (!IsNaN(m_dLastReturnedValue))
		return m_dLastReturnedValue;
	return GetValue(NORMAL_TEMPERATURE, NORMAL_PRESSURE);
}

size_t CTPProperty::CorrelationsNumber() const
{
	return m_vCorrelations.size();
}

ECorrelationTypes CTPProperty::GetCorrelationType(size_t _nIndex) const
{
	if (_nIndex < m_vCorrelations.size())
		return m_vCorrelations[_nIndex]->GetType();
	return ECorrelationTypes::UNDEFINED;
}

CCorrelation CTPProperty::GetCorrelation(double _dT, double _dP) const
{
	for (const auto& correlation : m_vCorrelations)
		if (correlation->IsInInterval(_dT, _dP))
			return *correlation;
	return CCorrelation();
}

CCorrelation CTPProperty::GetCorrelation(size_t _nIndex) const
{
	if (_nIndex < m_vCorrelations.size())
		return *m_vCorrelations[_nIndex];
	return CCorrelation();
}

size_t CTPProperty::AddCorrelation()
{
	return AddCorrelation(DEFAULT_T1, DEFAULT_T2, DEFAULT_P1, DEFAULT_P2, ECorrelationTypes::CONSTANT, std::vector<double>(1, DEFAULT_CORR_VALUE));
}

size_t CTPProperty::AddCorrelation(double _dConstValue)
{
	return AddCorrelation(DEFAULT_T1, DEFAULT_T2, DEFAULT_P1, DEFAULT_P2, ECorrelationTypes::CONSTANT, std::vector<double>(1, _dConstValue));
}

size_t CTPProperty::AddCorrelation(double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams)
{
	std::unique_ptr<CCorrelation> pCorr(new CCorrelation(_dT1, _dT2, _dP1, _dP2, _nType, _vParams));
	m_vCorrelations.push_back(std::move(pCorr));
	ResetConditions();
	return m_vCorrelations.size() - 1;
}

void CTPProperty::SetCorrelation(size_t _nIndex, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams)
{
	if (_nIndex < m_vCorrelations.size())
		if (m_vCorrelations[_nIndex]->SetCorrelation(_dT1, _dT2, _dP1, _dP2, _nType, _vParams))
			ResetConditions();
}

void CTPProperty::SetCorrelation(double _dConstValue)
{
	RemoveAllCorrelations();
	AddCorrelation(_dConstValue);
}

void CTPProperty::RemoveCorrelation(size_t _nIndex)
{
	if (_nIndex < m_vCorrelations.size())
	{
		m_vCorrelations.erase(m_vCorrelations.begin() + _nIndex);
		ResetConditions();
	}
}

void CTPProperty::RemoveAllCorrelations()
{
	m_vCorrelations.clear();
	ResetConditions();
}

void CTPProperty::InitializeConditions(double _dT, double _dP) const
{
	(void)GetValue(_dT, _dP);
}

void CTPProperty::SaveToProtobuf(ProtoTPProperty& _protoTPProperty)
{
	_protoTPProperty.set_propertytype(m_nPropertyType);
	for (auto& correlation : m_vCorrelations)
		correlation->SaveToProtobuf(*_protoTPProperty.add_correlation());
}

void CTPProperty::LoadFromProtobuf(const ProtoTPProperty& _protoTPProperty)
{
	m_vCorrelations.clear();
	for (int i = 0; i < _protoTPProperty.correlation_size(); ++i)
	{
		m_vCorrelations.push_back(std::make_unique<CCorrelation>());
		m_vCorrelations.back()->LoadFromProtobuf(_protoTPProperty.correlation(i));
	}
}

void CTPProperty::ResetConditions() const
{
	m_nLastUsedCorrelation = -1;
	m_dLastReturnedValue = _NOT_A_NUMBER;
}
