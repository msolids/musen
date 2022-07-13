/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MaterialsDatabase.h"
#include "MUSENStringFunctions.h"
#include "DisableWarningHelper.h"
#include <cmath>
#include <cfloat>
#include <fstream>
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "GeneratedFiles/MaterialsDatabase.pb.h"
PRAGMA_WARNING_POP

CMaterialsDatabase::CMaterialsDatabase()
{
	m_sFileName = DEFAULT_MDB_FILE_NAME;
}

CMaterialsDatabase::~CMaterialsDatabase()
{
	ClearData();
}

CMaterialsDatabase& CMaterialsDatabase::operator=(const CMaterialsDatabase& _mdb)
{
	if (this != &_mdb)
	{
		ClearData();
		m_sFileName = _mdb.m_sFileName;
		for (auto c : _mdb.m_vCompounds)		m_vCompounds.push_back(new CCompound(*c));
		for (auto i : _mdb.m_vInteractions)		m_vInteractions.push_back(new CInteraction(*i));
		for (auto m : _mdb.m_vMixtures)			m_vMixtures.push_back(new CMixture(*m));
	}
	return *this;
}

std::string CMaterialsDatabase::GetFileName() const
{
	return m_sFileName;
}

void CMaterialsDatabase::CreateNewDatabase()
{
	m_sFileName.clear();
	ClearData();
}

void CMaterialsDatabase::InitializeConditions(double _dT, double _dP)
{
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		m_vCompounds[i]->InitializeConditions(_dT, _dP);
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		m_vInteractions[i]->InitializeConditions(_dT, _dP);
}

void CMaterialsDatabase::ClearData()
{
	for (auto c : m_vCompounds)		delete c;
	for (auto i : m_vInteractions)	delete i;
	for (auto m : m_vMixtures)		delete m;
	m_vCompounds.clear();
	m_vInteractions.clear();
	m_vMixtures.clear();
}

void CMaterialsDatabase::SaveToFile(const std::string& _sFileName /*= "" */)
{
	if (!_sFileName.empty()) m_sFileName = _sFileName;

	ProtoMaterialsDatabase protoMDB;
	SaveToProtobufFile(protoMDB);

	// Ok Google. Save my file.
	std::fstream outFile(UnicodePath(m_sFileName), std::ios::out | std::ios::trunc | std::ios::binary);
	std::string data;
	// TODO: consider to use SerializeToZeroCopyStream() for performance
	protoMDB.SerializeToString(&data);
	outFile << data;
	outFile.close();
}

void CMaterialsDatabase::LoadFromFile(const std::string& _sFileName /*= "" */)
{
	std::string sNewFile = _sFileName.empty() ? DEFAULT_MDB_FILE_NAME : _sFileName;

	std::fstream inputFile(UnicodePath(sNewFile), std::ios::in | std::ios::binary);
	if (!inputFile) return;

	m_sFileName = sNewFile;

	// Ok Google. Load my file.
	ProtoMaterialsDatabase protoMDB;
	// TODO: consider to use ParseFromZeroCopyStream() for performance
	if (!protoMDB.ParseFromString(std::string(std::istreambuf_iterator<char>(inputFile), std::istreambuf_iterator<char>())))
		return;

	ClearData();
	LoadFromProtobufFile(protoMDB);
}

void CMaterialsDatabase::SaveToProtobufFile(ProtoMaterialsDatabase& _protoMDB)
{
	_protoMDB.clear_mixture();
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		m_vMixtures[i]->SaveToProtobuf(*_protoMDB.add_mixture());
	_protoMDB.clear_interaction();
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		m_vInteractions[i]->SaveToProtobuf(*_protoMDB.add_interaction());
	_protoMDB.clear_compound();
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		m_vCompounds[i]->SaveToProtobuf(*_protoMDB.add_compound());
}

void CMaterialsDatabase::LoadFromProtobufFile(const ProtoMaterialsDatabase& _protoMDB)
{
	ClearData();
	for (int i = 0; i < _protoMDB.mixture_size(); ++i)
	{
		m_vMixtures.push_back(new CMixture());
		m_vMixtures.back()->LoadFromProtobuf(_protoMDB.mixture(i));
	}
	for (int i = 0; i < _protoMDB.interaction_size(); ++i)
	{
		m_vInteractions.push_back(new CInteraction("", ""));
		m_vInteractions.back()->LoadFromProtobuf(_protoMDB.interaction(i));
	}
	for (int i = 0; i < _protoMDB.compound_size(); ++i)
	{
		m_vCompounds.push_back(new CCompound());
		m_vCompounds.back()->LoadFromProtobuf(_protoMDB.compound(i));
	}
}

size_t CMaterialsDatabase::CompoundsNumber() const
{
	return m_vCompounds.size();
}

CCompound* CMaterialsDatabase::AddCompound(const std::string& _sCompoundKey /*= "" */)
{
	// confirm the uniqueness of the key
	std::string sKey = GenerateUniqueKey(_sCompoundKey, GetCompoundsKeys());
	// add new compound
	m_vCompounds.push_back(new CCompound(sKey));
	// add corresponding interactions
	ConformInteractionsAdd(sKey);
	return m_vCompounds.back();
}

CCompound* CMaterialsDatabase::AddCompound(const CCompound& _compound)
{
	// generate unique key
	std::string sKey = GenerateUniqueKey(_compound.GetKey(), GetCompoundsKeys());
	// add new compound
	m_vCompounds.push_back(new CCompound(_compound));
	// set key
	m_vCompounds.back()->SetKey(sKey);
	// add corresponding interactions
	ConformInteractionsAdd(m_vCompounds.back()->GetKey());
	return m_vCompounds.back();
}

CCompound* CMaterialsDatabase::GetCompound(size_t _iCompound)
{
	return const_cast<CCompound*>(static_cast<const CMaterialsDatabase&>(*this).GetCompound(_iCompound));
}

const CCompound* CMaterialsDatabase::GetCompound(size_t _iCompound) const
{
	if (_iCompound < m_vCompounds.size())
		return m_vCompounds[_iCompound];
	return nullptr;
}

CCompound* CMaterialsDatabase::GetCompoundByName(const std::string& _sCompoundName)
{
	return const_cast<CCompound*>(static_cast<const CMaterialsDatabase&>(*this).GetCompoundByName(_sCompoundName));
}

const CCompound* CMaterialsDatabase::GetCompoundByName(const std::string& _sCompoundName) const
{
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		if (m_vCompounds[i]->GetName() == _sCompoundName)
			return m_vCompounds[i];
	return nullptr;
}

CCompound* CMaterialsDatabase::GetCompound(const std::string& _sCompoundKey)
{
	return const_cast<CCompound*>(static_cast<const CMaterialsDatabase&>(*this).GetCompound(_sCompoundKey));
}

const CCompound* CMaterialsDatabase::GetCompound(const std::string& _sCompoundKey) const
{
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		if (m_vCompounds[i]->GetKey() == _sCompoundKey)
			return m_vCompounds[i];
	return nullptr;
}

std::vector<CCompound*> CMaterialsDatabase::GetCompounds()
{
	return m_vCompounds;
}

std::vector<const CCompound*> CMaterialsDatabase::GetCompounds() const
{
	std::vector<const CCompound*> res;
	res.reserve(m_vCompounds.size());
	for (const auto& c : m_vCompounds)
		res.push_back(c);
	return res;
}

int CMaterialsDatabase::GetCompoundIndex(const std::string& _sCompoundKey) const
{
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		if (m_vCompounds[i]->GetKey() == _sCompoundKey)
			return static_cast<int>(i);
	return -1;
}

std::string CMaterialsDatabase::GetCompoundKey(size_t _iCompound) const
{
	if (_iCompound < m_vCompounds.size())
		return m_vCompounds[_iCompound]->GetKey();
	return "";
}

std::vector<std::string> CMaterialsDatabase::GetCompoundsKeys() const
{
	std::vector<std::string> keys;
	for (const auto& compound : m_vCompounds)
		keys.push_back(compound->GetKey());
	return keys;
}

std::string CMaterialsDatabase::GetCompoundName(size_t _iCompound) const
{
	if (_iCompound < m_vCompounds.size())
		return m_vCompounds[_iCompound]->GetName();
	return "";
}

std::string CMaterialsDatabase::GetCompoundName(const std::string& _sCompoundKey) const
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		return GetCompoundName(nIndex);
	return "";
}

std::vector<std::string> CMaterialsDatabase::GetCompoundsNames() const
{
	std::vector<std::string> names;
	for (const auto& compound : m_vCompounds)
		names.push_back(compound->GetName());
	return names;
}

void CMaterialsDatabase::RemoveCompound(size_t _iCompound)
{
	if (_iCompound < m_vCompounds.size())
	{
		ConformInteractionsRemove(m_vCompounds[_iCompound]->GetKey());
		ConformMixturesRemove(m_vCompounds[_iCompound]->GetKey());
		delete m_vCompounds[_iCompound];
		m_vCompounds.erase(m_vCompounds.begin() + _iCompound);
	}
}

void CMaterialsDatabase::RemoveCompound(const std::string& _sCompoundKey)
{
	int index = GetCompoundIndex(_sCompoundKey);
	if (index != -1)
		RemoveCompound(index);
}

void CMaterialsDatabase::UpCompound(size_t _iCompound)
{
	if (_iCompound < m_vCompounds.size() && _iCompound != 0)
		std::iter_swap(m_vCompounds.begin() + _iCompound, m_vCompounds.begin() + _iCompound - 1);
}

void CMaterialsDatabase::DownCompound(size_t _iCompound)
{
	if ((_iCompound < m_vCompounds.size()) && (_iCompound != (m_vCompounds.size() - 1)))
		std::iter_swap(m_vCompounds.begin() + _iCompound, m_vCompounds.begin() + _iCompound + 1);
}

double CMaterialsDatabase::GetConstPropertyValue(const std::string& _sCompoundKey, unsigned _nConstPropType) const
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		return GetConstPropertyValue(nIndex, _nConstPropType);
	return _NOT_A_NUMBER;
}

double CMaterialsDatabase::GetConstPropertyValue(size_t _iCompound, unsigned _nConstPropType) const
{
	if (const CCompound *comp = GetCompound(_iCompound))
		return comp->GetConstPropertyValue(_nConstPropType);
	return _NOT_A_NUMBER;
}

double CMaterialsDatabase::GetTPPropertyValue(const std::string& _sCompoundKey, unsigned _nTPPropType, double _dT, double _dP) const
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		return GetTPPropertyValue(nIndex, _nTPPropType, _dT, _dP);
	return _NOT_A_NUMBER;
}

double CMaterialsDatabase::GetTPPropertyValue(size_t _iCompound, unsigned _nTPPropType, double _dT, double _dP) const
{
	if (const CCompound *comp = GetCompound(_iCompound))
		return comp->GetTPPropertyValue(_nTPPropType, _dT, _dP);
	return _NOT_A_NUMBER;
}

double CMaterialsDatabase::GetPropertyValue(const std::string& _sCompoundKey, unsigned _nPropType) const
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		return GetPropertyValue(nIndex, _nPropType);
	return _NOT_A_NUMBER;
}

double CMaterialsDatabase::GetPropertyValue(size_t _iCompound, unsigned _nPropType) const
{
	if (const CCompound *comp = GetCompound(_iCompound))
		return comp->GetPropertyValue(_nPropType);
	return _NOT_A_NUMBER;
}

void CMaterialsDatabase::SetConstPropertyValue(const std::string& _sCompoundKey, unsigned _nConstPropType, double _dValue)
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		SetConstPropertyValue(nIndex, _nConstPropType, _dValue);
}

void CMaterialsDatabase::SetConstPropertyValue(size_t _iCompound, unsigned _nConstPropType, double _dValue)
{
	if (CCompound *comp = GetCompound(_iCompound))
		comp->SetConstPropertyValue(_nConstPropType, _dValue);
}

void CMaterialsDatabase::SetTPPropertyCorrelation(const std::string& _sCompoundKey, unsigned _nTPPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams)
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		SetTPPropertyCorrelation(nIndex, _nTPPropType, _dT1, _dT2, _dP1, _dP2, _nCorrType, _vParams);
}

void CMaterialsDatabase::SetTPPropertyCorrelation(size_t _iCompound, unsigned _nTPPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams)
{
	if (CCompound *comp = GetCompound(_iCompound))
		comp->SetTPPropertyCorrelation(_nTPPropType, _dT1, _dT2, _dP1, _dP2, _nCorrType, _vParams);
}

void CMaterialsDatabase::SetPropertyValue(const std::string& _sCompoundKey, unsigned _nPropType, double _dValue)
{
	int nIndex = GetCompoundIndex(_sCompoundKey);
	if (nIndex != -1)
		SetPropertyValue(nIndex, _nPropType, _dValue);
}

void CMaterialsDatabase::SetPropertyValue(size_t _iCompound, unsigned _nPropType, double _dValue)
{
	if (CCompound *comp = GetCompound(_iCompound))
		comp->SetPropertyValue(_nPropType, _dValue);
}

size_t CMaterialsDatabase::InteractionsNumber() const
{
	return m_vInteractions.size();
}

CInteraction* CMaterialsDatabase::GetInteraction(size_t _iInteraction)
{
	return const_cast<CInteraction*>(static_cast<const CMaterialsDatabase&>(*this).GetInteraction(_iInteraction));
}

const CInteraction* CMaterialsDatabase::GetInteraction(size_t _iInteraction) const
{
	if (_iInteraction < m_vInteractions.size())
		return m_vInteractions[_iInteraction];
	return nullptr;
}

CInteraction* CMaterialsDatabase::GetInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2)
{
	return const_cast<CInteraction*>(static_cast<const CMaterialsDatabase&>(*this).GetInteraction(_sCompoundKey1, _sCompoundKey2));
}

const CInteraction* CMaterialsDatabase::GetInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2) const
{
	int index = GetInteractionIndex(_sCompoundKey1, _sCompoundKey2);
	if (index != -1)
		return m_vInteractions[index];
	return nullptr;
}

std::vector<CInteraction*> CMaterialsDatabase::GetInteractions()
{
	return m_vInteractions;
}

std::vector<const CInteraction*> CMaterialsDatabase::GetInteractions() const
{
	std::vector<const CInteraction*> res;
	res.reserve(m_vInteractions.size());
	for (const auto& i : m_vInteractions)
		res.push_back(i);
	return res;
}

int CMaterialsDatabase::GetInteractionIndex(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2) const
{
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		if (m_vInteractions[i]->IsBetween(_sCompoundKey1, _sCompoundKey2))
			return static_cast<int>(i);
	return -1;
}

double CMaterialsDatabase::GetInteractionValue(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nIntType, double _dT, double _dP) const
{
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		if (m_vInteractions[i]->IsBetween(_sCompKey1, _sCompKey2))
			return m_vInteractions[i]->GetTPPropertyValue(_nIntType, _dT, _dP);
	return _NOT_A_NUMBER;
}

double CMaterialsDatabase::GetInteractionValue(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nIntType) const
{
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		if (m_vInteractions[i]->IsBetween(_sCompKey1, _sCompKey2))
			return m_vInteractions[i]->GetPropertyValue(_nIntType);
	return _NOT_A_NUMBER;
}

void CMaterialsDatabase::SetInteractionCorrelation(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams)
{
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		if (m_vInteractions[i]->IsBetween(_sCompKey1, _sCompKey2))
			m_vInteractions[i]->SetTPPropertyCorrelation(_nPropType, _dT1, _dT2, _dP1, _dP2, _nCorrType, _vParams);
}

void CMaterialsDatabase::SetInteractionValue(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nPropType, double _dValue)
{
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		if (m_vInteractions[i]->IsBetween(_sCompKey1, _sCompKey2))
			m_vInteractions[i]->SetPropertyValue(_nPropType, _dValue);
}

size_t CMaterialsDatabase::MixturesNumber() const
{
	return m_vMixtures.size();
}

CMixture* CMaterialsDatabase::AddMixture(const std::string& _sMixtureKey /*= "" */)
{
	// confirm the uniqueness of the key
	std::string sKey = GenerateUniqueKey(_sMixtureKey, GetMixturesKeys());
	// add new mixture
	m_vMixtures.push_back(new CMixture(sKey));
	return m_vMixtures.back();
}

CMixture* CMaterialsDatabase::AddMixture(const CMixture& _mixture)
{
	// generate unique key
	std::string sKey = GenerateUniqueKey(_mixture.GetKey(), GetMixturesKeys());
	// add new mixture
	m_vMixtures.push_back(new CMixture(_mixture));
	// set key
	m_vMixtures.back()->SetKey(sKey);
	return m_vMixtures.back();
}

int CMaterialsDatabase::GetMixtureIndex(const std::string& _sMixtureKey) const
{
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		if (m_vMixtures[i]->GetKey() == _sMixtureKey)
			return static_cast<int>(i);
	return -1;
}

std::string CMaterialsDatabase::GetMixtureName(size_t _iMixture) const
{
	if (_iMixture < m_vMixtures.size())
		return m_vMixtures[_iMixture]->GetName();
	return "";
}

std::string CMaterialsDatabase::GetMixtureName(const std::string& _sMixtureKey) const
{
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		if (m_vMixtures[i]->GetKey() == _sMixtureKey)
			return GetMixtureName(i);
	return "";
}

std::string CMaterialsDatabase::GetMixtureKey(size_t _iMixture) const
{
	if (_iMixture < m_vMixtures.size())
		return m_vMixtures[_iMixture]->GetKey();
	return "";
}

CMixture* CMaterialsDatabase::GetMixture(size_t _iMixture)
{
	return const_cast<CMixture*>(static_cast<const CMaterialsDatabase&>(*this).GetMixture(_iMixture));
}

const CMixture* CMaterialsDatabase::GetMixture(size_t _iMixture) const
{
	if (_iMixture < m_vMixtures.size())
		return m_vMixtures[_iMixture];
	return nullptr;
}

CMixture* CMaterialsDatabase::GetMixture(const std::string& _sMixtureKey)
{
	return const_cast<CMixture*>(static_cast<const CMaterialsDatabase&>(*this).GetMixture(_sMixtureKey));
}

const CMixture* CMaterialsDatabase::GetMixture(const std::string& _sMixtureKey) const
{
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		if (m_vMixtures[i]->GetKey() == _sMixtureKey)
			return m_vMixtures[i];
	return nullptr;
}

std::vector<CMixture*> CMaterialsDatabase::GetMixtures()
{
	return m_vMixtures;
}

std::vector<const CMixture*> CMaterialsDatabase::GetMixtures() const
{
	std::vector<const CMixture*> res;
	res.reserve(m_vMixtures.size());
	for (const auto& m : m_vMixtures)
		res.push_back(m);
	return res;
}

CMixture* CMaterialsDatabase::GetMixtureByName(const std::string& _mixtureName)
{
	return const_cast<CMixture*>(static_cast<const CMaterialsDatabase&>(*this).GetMixtureByName(_mixtureName));
}

const CMixture* CMaterialsDatabase::GetMixtureByName(const std::string& _mixtureName) const
{
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		if (m_vMixtures[i]->GetName() == _mixtureName)
			return m_vMixtures[i];
	return nullptr;
}

void CMaterialsDatabase::RemoveMixture(size_t _iMixture)
{
	if (_iMixture < m_vMixtures.size())
	{
		delete m_vMixtures[_iMixture];
		m_vMixtures.erase(m_vMixtures.begin() + _iMixture);
	}
}

void CMaterialsDatabase::RemoveMixture(const std::string& _sMixtureKey)
{
	int index = GetMixtureIndex(_sMixtureKey);
	if (index != -1)
		RemoveMixture(index);
}

std::string CMaterialsDatabase::IsMixtureCorrect(const std::string& _sMixtureKey) const
{
	const CMixture* pMixture = GetMixture(_sMixtureKey);
	if (!pMixture)
		return "No correct mixture was specified" ;
	if (pMixture->FractionsNumber() == 0)
		return "No fractions has been specified for mixture '" + pMixture->GetName() + "'";

	// check that all compounds was defined in the database
	for (unsigned j = 0; j < pMixture->FractionsNumber(); ++j)
		if (!GetCompound(pMixture->GetFractionCompound(j)))
			return "Some fractions was wrong specified for mixture '" + pMixture->GetName() + "'";

	double dTotalFraction = 0;
	for (unsigned j = 0; j < pMixture->FractionsNumber(); ++j)
		dTotalFraction += pMixture->GetFractionValue(j);
	if (fabs(dTotalFraction - 1) > 16 * DBL_EPSILON)
		return "The total fraction of material (" + pMixture->GetName() + ") is not equal to 1 (" + std::to_string(dTotalFraction) + ")";
	return "";
}

void CMaterialsDatabase::UpMixture(size_t _iMixture)
{
	if (_iMixture < m_vMixtures.size() && _iMixture != 0)
		std::iter_swap(m_vMixtures.begin() + _iMixture, m_vMixtures.begin() + _iMixture - 1);
}

void CMaterialsDatabase::DownMixture(size_t _iMixture)
{
	if ((_iMixture < m_vMixtures.size()) && (_iMixture != (m_vMixtures.size() - 1)))
		std::iter_swap(m_vMixtures.begin() + _iMixture, m_vMixtures.begin() + _iMixture + 1);
}

std::string CMaterialsDatabase::IsDataCorrect() const
{
	for (const auto& c : m_vCompounds)
	{
		if (c->GetPropertyValue(PROPERTY_DENSITY) < 0.0)
			return "Negative density in material '" + c->GetName() + "' is not allowed.";
		if (c->GetPropertyValue(PROPERTY_THERMAL_CONDUCTIVITY) == 0.0)
			return "Zero thermal conductivity in material '" + c->GetName() + "' is not allowed.";
		if (c->GetPropertyValue(PROPERTY_YOUNG_MODULUS) == 0.0)
			return "Zero Young modulus in material '" + c->GetName() + "' is not allowed.";
		if (!IsInRange(c->GetPropertyValue(PROPERTY_POISSON_RATIO), 0.0, 0.5))
			return "Poisson ratio outside the range [0..0.5] in material '" + c->GetName() + "' is not allowed.";
		if (c->GetPropertyValue(PROPERTY_SURFACE_TENSION) < 0.0)
			return "Negative surface tension in material '" + c->GetName() + "' is not allowed.";
	}

	return "";
}

CInteraction* CMaterialsDatabase::AddInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2)
{
	// check existence
	for (size_t i = 0; i < m_vInteractions.size(); ++i)
		if (m_vInteractions[i]->IsBetween(_sCompoundKey1, _sCompoundKey2))
			return m_vInteractions[i];

	// create new interaction
	m_vInteractions.push_back(new CInteraction(_sCompoundKey1, _sCompoundKey2));
	return m_vInteractions.back();
}

void CMaterialsDatabase::RemoveInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2)
{
	int index = GetInteractionIndex(_sCompoundKey1, _sCompoundKey2);
	if (index != -1)
		RemoveInteraction(index);
}

void CMaterialsDatabase::RemoveInteraction(size_t _iInteraction)
{
	if (_iInteraction < m_vInteractions.size())
	{
		delete m_vInteractions[_iInteraction];
		m_vInteractions.erase(m_vInteractions.begin() + _iInteraction);
	}
}

void CMaterialsDatabase::ConformInteractions()
{
	// add necessary default interactions
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		for (size_t j = 0; j < m_vCompounds.size(); ++j)
			if (!GetInteraction(m_vCompounds[i]->GetKey(), m_vCompounds[j]->GetKey()))
				AddInteraction(m_vCompounds[i]->GetKey(), m_vCompounds[j]->GetKey());

	// remove old unnecessary interactions
	unsigned i = 0;
	while (i < m_vInteractions.size())
		if ((!GetCompound(m_vInteractions[i]->GetKey1())) || (!GetCompound(m_vInteractions[i]->GetKey2())))
			RemoveInteraction(i);
		else
			i++;
}

void CMaterialsDatabase::ConformInteractionsAdd(const std::string& _sCompoundKey)
{
	// add necessary default interactions
	for (size_t i = 0; i < m_vCompounds.size(); ++i)
		AddInteraction(_sCompoundKey, m_vCompounds[i]->GetKey());
}

void CMaterialsDatabase::ConformInteractionsRemove(const std::string& _sCompoundKey)
{
	// remove unnecessary interactions
	unsigned i = 0;
	while (i < m_vInteractions.size())
		if ((m_vInteractions[i]->GetKey1() == _sCompoundKey) || (m_vInteractions[i]->GetKey2() == _sCompoundKey))
			RemoveInteraction(i);
		else
			i++;
}

void CMaterialsDatabase::ConformMixturesRemove(const std::string& _sCompoundKey)
{
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		for (size_t j = 0; j < m_vMixtures[i]->FractionsNumber(); ++j)
			if (m_vMixtures[i]->GetFractionCompound(j) == _sCompoundKey)
				m_vMixtures[i]->SetFractionCompound(j, "");
}

std::vector<std::string> CMaterialsDatabase::GetMixturesKeys() const
{
	std::vector<std::string> vKeys;
	for (size_t i = 0; i < m_vMixtures.size(); ++i)
		vKeys.push_back(m_vMixtures[i]->GetKey());
	return vKeys;
}