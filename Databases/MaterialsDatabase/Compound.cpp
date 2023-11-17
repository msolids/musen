/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Compound.h"
#include "MUSENStringFunctions.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "MaterialsDatabase.pb.h"
PRAGMA_WARNING_POP

CCompound::CCompound(const std::string& _sKey /*= "" */)
{
	// set basic data
	m_sUniqueKey = _sKey.empty() ? GenerateKey() : _sKey;
	m_sCompoundName = COMPOUND_UNDEFINED_NAME;
	m_sAuthorName = COMPOUND_UNDEFINED_AUTHOR;

	m_Color.SetColor(0, 0, 1, 1); 	// set default object color
	m_CreationDate = CurrentDate(); // set creation date
	m_nPhaseState = _PHASE_TYPE_DEFAULT; // set phase

	// create default constant properties
	unsigned constTypes[] = _CONST_PROPERTIES;
	std::string constNames[] = _CONST_PROPERTIES_NAMES;
	std::string constUnits[] = _CONST_PROPERTIES_UNITS;
	double constDefaults[] = _CONST_PROPERTIES_DEFAULTS;
	for (size_t i = 0; i < _CONST_PROPERTIES_NUMBER; ++i)
	{
		m_vConstProperties.push_back(new CConstProperty(constTypes[i], constNames[i], constUnits[i]));
		m_vConstProperties.back()->SetValue(constDefaults[i]);
	}

	//create default TPProperties
	unsigned tpTypes[] = _TP_PROPERTIES;
	std::string tpNames[] = _TP_PROPERTIES_NAMES;
	std::string tpUnits[] = _TP_PROPERTIES_UNITS;
	double tpDefaults[] = _TP_PROPERTIES_DEFAULTS;
	for (size_t i = 0; i < _TP_PROPERTIES_NUMBER; ++i)
	{
		m_vProperties.push_back(new CTPProperty(tpTypes[i], tpNames[i], tpUnits[i]));
		m_vProperties.back()->AddCorrelation(tpDefaults[i]);
	}
}

CCompound::CCompound(const CCompound& _compound)
{
	Copy(_compound);
}

CCompound::~CCompound()
{
	Clear();
}

CCompound& CCompound::operator=(const CCompound& _compound)
{
	if (this != &_compound)
	{
		Clear();
		Copy(_compound);
	}
	return *this;
}

void CCompound::Clear()
{
	for (size_t i = 0; i < m_vConstProperties.size(); ++i)
		delete m_vConstProperties[i];
	m_vConstProperties.clear();
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		delete m_vProperties[i];
	m_vProperties.clear();
}

void CCompound::Copy(const CCompound& _compound)
{
	// set basic data
	m_sUniqueKey = _compound.m_sUniqueKey;
	m_sCompoundName = _compound.m_sCompoundName;
	m_sAuthorName = _compound.m_sAuthorName;

	m_Color = _compound.m_Color; // set default object color
	m_CreationDate = CurrentDate(); 	// set creation date
	m_nPhaseState = _compound.m_nPhaseState; 	// set phase

	// create default constant properties
	for (auto c : _compound.m_vConstProperties)
		m_vConstProperties.push_back(new CConstProperty(*c));

	//create default TPProperties
	for (auto tp : _compound.m_vProperties)
		m_vProperties.push_back(new CTPProperty(*tp));
}

std::string CCompound::GetName() const
{
	return m_sCompoundName;
}

void CCompound::SetName(const std::string& _sName)
{
	m_sCompoundName = _sName;
}

std::string CCompound::GetKey() const
{
	return m_sUniqueKey;
}

void CCompound::SetKey(const std::string& _sKey)
{
	m_sUniqueKey = _sKey;
}

std::string CCompound::GetAuthorName() const
{
	return m_sAuthorName;
}

void CCompound::SetAuthorName(const std::string& _sAuthorName)
{
	m_sAuthorName = _sAuthorName;
}

CColor CCompound::GetColor() const
{
	return m_Color;
}

void CCompound::SetColor(CColor& _color)
{
	m_Color = _color;
}

void CCompound::SetColor(float _r, float _g, float _b, float _a /*= 1.0 */)
{
	m_Color.r = _r;
	m_Color.g = _g;
	m_Color.b = _b;
	m_Color.a = _a;
}

SDate CCompound::GetCreationDate() const
{
	return m_CreationDate;
}

void CCompound::SetCreationDate(const SDate& _date)
{
	m_CreationDate = _date;
}

void CCompound::SetCreationDate(unsigned _nY, unsigned _nM, unsigned _nD)
{
	m_CreationDate.nYear = _nY;
	m_CreationDate.nMonth = _nM;
	m_CreationDate.nDay = _nD;
}

size_t CCompound::TPPropertiesNumber() const
{
	return m_vProperties.size();
}

size_t CCompound::ConstPropertiesNumber() const
{
	return m_vConstProperties.size();
}

unsigned CCompound::GetPhaseState() const
{
	return m_nPhaseState;
}

void CCompound::SetPhaseState(unsigned _nState)
{
	m_nPhaseState = _nState;
}

CTPProperty* CCompound::GetProperty(unsigned _nType)
{
	return const_cast<CTPProperty*>(static_cast<const CCompound&>(*this).GetProperty(_nType));
}

const CTPProperty* CCompound::GetProperty(unsigned _nType) const
{
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		if (m_vProperties[i]->GetType() == _nType)
			return m_vProperties[i];
	return nullptr;
}

CTPProperty* CCompound::GetPropertyByIndex(unsigned _nIndex)
{
	return const_cast<CTPProperty*>(static_cast<const CCompound&>(*this).GetPropertyByIndex(_nIndex));
}

const CTPProperty* CCompound::GetPropertyByIndex(unsigned _nIndex) const
{
	if (_nIndex < m_vProperties.size())
		return m_vProperties[_nIndex];
	return nullptr;
}

CConstProperty* CCompound::GetConstProperty(unsigned _nType)
{
	return const_cast<CConstProperty*>(static_cast<const CCompound&>(*this).GetConstProperty(_nType));
}

const CConstProperty* CCompound::GetConstProperty(unsigned _nType) const
{
	for (size_t i = 0; i < m_vConstProperties.size(); ++i)
		if (m_vConstProperties[i]->GetType() == _nType)
			return m_vConstProperties[i];
	return nullptr;
}

CConstProperty* CCompound::GetConstPropertyByIndex(unsigned _nIndex)
{
	return const_cast<CConstProperty*>(static_cast<const CCompound&>(*this).GetConstPropertyByIndex(_nIndex));
}

const CConstProperty* CCompound::GetConstPropertyByIndex(unsigned _nIndex) const
{
	if (_nIndex < m_vConstProperties.size())
		return m_vConstProperties[_nIndex];
	return nullptr;
}

double CCompound::GetConstPropertyValue(unsigned _nType) const
{
	if (const CConstProperty *prop = GetConstProperty(_nType))
		return prop->GetValue();
	return _NOT_A_NUMBER;
}

double CCompound::GetTPPropertyValue(unsigned _nType, double _dT, double _dP) const
{
	if (const CTPProperty *prop = GetProperty(_nType))
		return prop->GetValue(_dT, _dP);
	return _NOT_A_NUMBER;
}

double CCompound::GetPropertyValue(unsigned _nType) const
{
	if (const CTPProperty *prop = GetProperty(_nType))
		return prop->GetValue();
	return GetConstPropertyValue(_nType);
}

void CCompound::SetConstPropertyValue(unsigned _nPropType, double _dValue)
{
	if (CConstProperty *prop = GetConstProperty(_nPropType))
		prop->SetValue(_dValue);
}

void CCompound::SetTPPropertyCorrelation(unsigned _nPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams)
{
	if (CTPProperty *prop = GetProperty(_nPropType))
		prop->AddCorrelation(_dT1, _dT2, _dP1, _dP2, _nCorrType, _vParams);
}

void CCompound::SetPropertyValue(unsigned _nPropType, double _dValue)
{
	if (CConstProperty *prop = GetConstProperty(_nPropType))
		prop->SetValue(_dValue);
	else if (CTPProperty *prop = GetProperty(_nPropType))
		prop->SetCorrelation(_dValue);
}

void CCompound::InitializeConditions(double _dT, double _dP)
{
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		m_vProperties[i]->InitializeConditions(_dT, _dP);
}

void CCompound::SaveToProtobuf(ProtoCompound& _protoCompound)
{
	_protoCompound.set_compoundname(m_sCompoundName);
	_protoCompound.set_uniquekey(m_sUniqueKey);
	_protoCompound.set_authorname(m_sAuthorName);
	ProtoDate *date = _protoCompound.mutable_creationdate();
	date->set_year(m_CreationDate.nYear);
	date->set_month(m_CreationDate.nMonth);
	date->set_day(m_CreationDate.nDay);
	ProtoColor *color = _protoCompound.mutable_color();
	color->set_r(m_Color.r);
	color->set_g(m_Color.g);
	color->set_b(m_Color.b);
	color->set_a(m_Color.a);
	_protoCompound.set_phase(m_nPhaseState);
	for (size_t i = 0; i < m_vConstProperties.size(); ++i)
		m_vConstProperties[i]->SaveToProtobuf(*_protoCompound.add_constproperty());
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		m_vProperties[i]->SaveToProtobuf(*_protoCompound.add_property());
}

void CCompound::LoadFromProtobuf(const ProtoCompound& _protoCompound)
{
	// load general data
	m_sCompoundName = _protoCompound.compoundname();
	m_sUniqueKey = _protoCompound.uniquekey();
	m_sAuthorName = _protoCompound.authorname();
	m_CreationDate.nYear = _protoCompound.creationdate().year();
	m_CreationDate.nMonth = _protoCompound.creationdate().month();
	m_CreationDate.nDay = _protoCompound.creationdate().day();
	m_Color.r = _protoCompound.color().r();
	m_Color.g = _protoCompound.color().g();
	m_Color.b = _protoCompound.color().b();
	m_Color.a = _protoCompound.color().a();
	m_nPhaseState = _protoCompound.phase();
	// load const properties
	for (size_t i = 0; i < m_vConstProperties.size(); ++i)
		for (int j = 0; j < _protoCompound.constproperty_size(); ++j)
			if (m_vConstProperties[i]->GetType() == _protoCompound.constproperty(j).propertytype())
			{
				m_vConstProperties[i]->LoadFromProtobuf(_protoCompound.constproperty(j));
				break;
			}

	// load TP properties
	for (size_t i = 0; i < m_vProperties.size(); ++i)
		for (int j = 0; j < _protoCompound.property_size(); ++j)
			if (m_vProperties[i]->GetType() == _protoCompound.property(j).propertytype())
			{
				m_vProperties[i]->LoadFromProtobuf(_protoCompound.property(j));
				break;
			}
}