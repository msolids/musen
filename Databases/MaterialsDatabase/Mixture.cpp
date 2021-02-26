/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Mixture.h"
#include "DefinesMDB.h"
#include "MUSENStringFunctions.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "GeneratedFiles/MaterialsDatabase.pb.h"
PRAGMA_WARNING_POP

SCompoundFraction::SCompoundFraction():
	sFractionName(COMPOUND_FRACTION_UNDEFINED_NAME), sCompoundKey(""), dDiameter(COMPOUND_FRACTION_DEFAULT_DIAMETER), dContactDiameter(COMPOUND_FRACTION_DEFAULT_DIAMETER), dFraction(0)
{
}

SCompoundFraction::SCompoundFraction(const std::string& _sName, const std::string& _sCompoundKey, double _dDiameter, double _dFraction):
	sFractionName(_sName), sCompoundKey(_sCompoundKey), dDiameter(_dDiameter), dContactDiameter(_dDiameter), dFraction(_dFraction)
{
}

SCompoundFraction::SCompoundFraction(const std::string& _sName, const std::string& _sCompoundKey, double _dDiameter, double _dContactDiameter, double _dFraction):
	sFractionName(_sName), sCompoundKey(_sCompoundKey), dDiameter(_dDiameter), dContactDiameter(_dContactDiameter), dFraction(_dFraction)
{
}

void SCompoundFraction::SaveToProtobuf(ProtoCompoundFraction& _protoFraction)
{
	_protoFraction.set_fractionname(sFractionName);
	_protoFraction.set_compoundkey(sCompoundKey);
	_protoFraction.set_diameter(dDiameter);
	_protoFraction.set_contact_diameter(dContactDiameter);
	_protoFraction.set_fraction(dFraction);
}

void SCompoundFraction::LoadFromProtobuf(const ProtoCompoundFraction& _protoFraction)
{
	sFractionName = _protoFraction.fractionname();
	sCompoundKey = _protoFraction.compoundkey();
	dDiameter = _protoFraction.diameter();
	if (_protoFraction.contact_diameter() != 0) // for old file formats //proto3 (in proto2 - has_contact_diameter)
		dContactDiameter = _protoFraction.contact_diameter();
	else
		dContactDiameter = dDiameter;
	dFraction = _protoFraction.fraction();
}

CMixture::CMixture(const std::string& _sKey /*= "" */)
{
	m_sUniqueKey = _sKey.empty() ? GenerateKey() : _sKey;
	m_sMixtureName = MIXTURE_UNDEFINED_NAME;
}

CMixture::CMixture(const std::string& _sKey, const std::string& _sName) :
	m_sUniqueKey(_sKey), m_sMixtureName(_sName)
{
}

CMixture::CMixture(const CMixture& _mixture)
{
	m_sUniqueKey = _mixture.m_sUniqueKey;
	m_sMixtureName = _mixture.m_sMixtureName;
	for (auto f : _mixture.m_vFractions)
		m_vFractions.push_back(new SCompoundFraction(*f));
}

CMixture::~CMixture()
{
	Clear();
}

std::string CMixture::GetName() const
{
	return m_sMixtureName;
}

void CMixture::SetName(const std::string& _sName)
{
	m_sMixtureName = _sName;
}

std::string CMixture::GetKey() const
{
	return m_sUniqueKey;
}

void CMixture::SetKey(const std::string& _sKey)
{
	m_sUniqueKey = _sKey;
}

void CMixture::Clear()
{
	for (size_t i = 0; i < m_vFractions.size(); ++i)
		delete m_vFractions[i];
	m_vFractions.clear();
}

size_t CMixture::FractionsNumber() const
{
	return m_vFractions.size();
}

size_t CMixture::AddFraction()
{
	m_vFractions.push_back(new SCompoundFraction());
	return m_vFractions.size() - 1;
}

size_t CMixture::AddFraction(const std::string& _sName, const std::string& _sCompoundKey, double _dDiameter, double _dFraction)
{
	m_vFractions.push_back(new SCompoundFraction(_sName, _sCompoundKey, _dDiameter, _dFraction));
	return m_vFractions.size() - 1;
}

size_t CMixture::AddFraction(const std::string& _sName, const std::string& _sCompoundKey, double _dDiameter, double _dContactDiameter, double _dFraction)
{
	m_vFractions.push_back(new SCompoundFraction(_sName, _sCompoundKey, _dDiameter, _dContactDiameter, _dFraction));
	return m_vFractions.size() - 1;
}

void CMixture::SetFractionName(size_t _iFraction, const std::string& _sName)
{
	if (_iFraction < m_vFractions.size())
		m_vFractions[_iFraction]->sFractionName = _sName;
}

void CMixture::SetFractionCompound(size_t _iFraction, const std::string& _sCompoundKey)
{
	if (_iFraction < m_vFractions.size())
		m_vFractions[_iFraction]->sCompoundKey = _sCompoundKey;
}

void CMixture::SetFractionDiameter(size_t _iFraction, double _dDiameter)
{
	if (_iFraction < m_vFractions.size())
		m_vFractions[_iFraction]->dDiameter = _dDiameter;
}

void CMixture::SetFractionContactDiameter(size_t _iFraction, double _dContactDiameter)
{
	if (_iFraction < m_vFractions.size())
		m_vFractions[_iFraction]->dContactDiameter = _dContactDiameter;
}

void CMixture::SetFractionValue(size_t _iFraction, double _dFraction)
{
	if (_iFraction < m_vFractions.size())
		m_vFractions[_iFraction]->dFraction = _dFraction;
}

std::string CMixture::GetFractionName(size_t _iFraction) const
{
	if (_iFraction < m_vFractions.size())
		return m_vFractions[_iFraction]->sFractionName;
	return "";
}

std::string CMixture::GetFractionCompound(size_t _iFraction) const
{
	if (_iFraction < m_vFractions.size())
		return m_vFractions[_iFraction]->sCompoundKey;
	return "";
}

double CMixture::GetFractionDiameter(size_t _iFraction) const
{
	if (_iFraction < m_vFractions.size())
		return m_vFractions[_iFraction]->dDiameter;
	return _NOT_A_NUMBER;
}

double CMixture::GetFractionContactDiameter(size_t _iFraction) const
{
	if (_iFraction < m_vFractions.size())
		return m_vFractions[_iFraction]->dContactDiameter;
	return _NOT_A_NUMBER;
}

double CMixture::GetMinFractionDiameter() const
{
	if (m_vFractions.empty()) return 0;
	double dMinDiameter = m_vFractions[0]->dDiameter;
	for(auto f : m_vFractions)
		if (f->dDiameter < dMinDiameter)
			dMinDiameter = f->dDiameter;
	return dMinDiameter;
}

double CMixture::GetMaxFractionDiameter() const
{
	if (m_vFractions.empty()) return 0;
	double dMaxDiameter = m_vFractions[0]->dDiameter;
	for (auto f : m_vFractions)
		if (f->dDiameter > dMaxDiameter)
			dMaxDiameter = f->dDiameter;
	return dMaxDiameter;
}

double CMixture::GetMinFractionContactDiameter() const
{
	if (m_vFractions.empty()) return 0;
	double dMinContactDiameter = m_vFractions[0]->dContactDiameter;
	for (auto f : m_vFractions)
		if (f->dContactDiameter < dMinContactDiameter)
			dMinContactDiameter = f->dContactDiameter;
	return dMinContactDiameter;
}

double CMixture::GetMaxFractionContactDiameter() const
{
	if (m_vFractions.empty()) return 0;
	double dMaxContactDiameter = m_vFractions[0]->dContactDiameter;
	for (auto f : m_vFractions)
		if (f->dContactDiameter > dMaxContactDiameter)
			dMaxContactDiameter = f->dContactDiameter;
	return dMaxContactDiameter;
}

double CMixture::GetFractionValue(size_t _iFraction) const
{
	if (_iFraction < m_vFractions.size())
		return m_vFractions[_iFraction]->dFraction;
	return 0;
}

void CMixture::RemoveFraction(size_t _iFraction)
{
	if (_iFraction < m_vFractions.size())
	{
		delete m_vFractions[_iFraction];
		m_vFractions.erase(m_vFractions.begin() + _iFraction);
	}
}

void CMixture::NormalizeFractions()
{
	double dTotalFraction = 0;
	for (size_t i = 0; i < m_vFractions.size(); ++i)
		dTotalFraction += m_vFractions[i]->dFraction;

	if (dTotalFraction == 0)
	{
		for (size_t i = 0; i < m_vFractions.size(); ++i)
		{
			m_vFractions[i]->dFraction = 1.0 / m_vFractions.size();
			dTotalFraction += m_vFractions[i]->dFraction;
		}
	}

	for (size_t i = 0; i < m_vFractions.size(); ++i)
		m_vFractions[i]->dFraction = m_vFractions[i]->dFraction / dTotalFraction;
}

void CMixture::UpFraction(size_t _iFraction)
{
	if (_iFraction < m_vFractions.size() && _iFraction != 0)
		std::iter_swap(m_vFractions.begin() + _iFraction, m_vFractions.begin() + _iFraction - 1);
}

void CMixture::DownFraction(size_t _iFraction)
{
	if ((_iFraction < m_vFractions.size()) && (_iFraction != (m_vFractions.size() - 1)))
		std::iter_swap(m_vFractions.begin() + _iFraction, m_vFractions.begin() + _iFraction + 1);
}

void CMixture::SaveToProtobuf(ProtoMixture& _protoMixture)
{
	_protoMixture.set_uniquekey(m_sUniqueKey);
	_protoMixture.set_mixturename(m_sMixtureName);
	for (size_t i = 0; i < m_vFractions.size(); ++i)
		m_vFractions[i]->SaveToProtobuf(*_protoMixture.add_fraction());
}

void CMixture::LoadFromProtobuf(const ProtoMixture& _protoMixture)
{
	m_sUniqueKey = _protoMixture.uniquekey();
	m_sMixtureName = _protoMixture.mixturename();
	for (int i = 0; i < _protoMixture.fraction_size(); ++i)
	{
		m_vFractions.push_back(new SCompoundFraction());
		m_vFractions.back()->LoadFromProtobuf(_protoMixture.fraction(i));
	}
}