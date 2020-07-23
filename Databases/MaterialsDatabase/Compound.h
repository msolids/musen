/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "TPProperty.h"
#include "ConstProperty.h"
#include "MixedFunctions.h"
#include "MUSENStringFunctions.h"

class CCompound
{
public:
	CCompound(const std::string& _sKey = "");
	CCompound(const CCompound& _compound);
	~CCompound();

	/// Copies data from other compound
	CCompound& operator=(const CCompound& _compound);

	/// Removes all properties from compound.
	void Clear();
	/// Copies data from another compound.
	void Copy(const CCompound& _compound);

	//////////////////////////////////////////////////////////////////////////
	/// Basic parameters

	/// Returns compound's name
	std::string GetName() const;
	/// Sets new compound's name
	void SetName(const std::string& _sName);

	/// Returns compound's unique key
	std::string GetKey() const;
	/// Sets new compound's key
	void SetKey(const std::string& _sKey);

	/// Returns compound's author name
	std::string GetAuthorName() const;
	/// Sets new compound's author name
	void SetAuthorName(const std::string& _sAuthorName);

	/// Returns current color of the compound
	CColor GetColor() const;
	/// Sets new color of the compound
	void SetColor(CColor& _color);
	/// Sets new color of the compound
	void SetColor(float _r, float _g, float _b, float _a = 1.0);

	/// Returns creation date
	SDate GetCreationDate() const;
	/// Sets new creation date of the compound
	void SetCreationDate(const SDate& _date);
	/// Sets new creation date of the compound
	void SetCreationDate(unsigned _nY, unsigned _nM, unsigned _nD);

	/// Returns number of TP properties
	size_t TPPropertiesNumber() const;
	/// Returns number of const properties
	size_t ConstPropertiesNumber() const;

	/// Returns phase state
	unsigned GetPhaseState() const;
	/// Sets new phase state
	void SetPhaseState(unsigned _nState);

	//////////////////////////////////////////////////////////////////////////
	/// Pointers getters

	/// Returns pointer to a specified TP property. Returns nullptr if property is not found.
	CTPProperty* GetProperty(unsigned _nType);
	/// Returns constant pointer to a specified TP property. Returns nullptr if property is not found.
	const CTPProperty* GetProperty(unsigned _nType) const;
	/// Returns pointer to a specified TP property by its index. Returns nullptr if property is not found.
	CTPProperty* GetPropertyByIndex(unsigned _nIndex);
	/// Returns constant pointer to a specified TP property by its index. Returns nullptr if property is not found.
	const CTPProperty* GetPropertyByIndex(unsigned _nIndex) const;
	/// Returns pointer to a specified const property. Returns nullptr if property is not found.
	CConstProperty* GetConstProperty(unsigned _nType);
	/// Returns constant pointer to a specified const property. Returns nullptr if property is not found.
	const CConstProperty* GetConstProperty(unsigned _nType) const;
	/// Returns pointer to a specified const property by its index. Returns nullptr if property is not found.
	CConstProperty* GetConstPropertyByIndex(unsigned _nIndex);
	/// Returns constant pointer to a specified const property by its index. Returns nullptr if property is not found.
	const CConstProperty* GetConstPropertyByIndex(unsigned _nIndex) const;

	//////////////////////////////////////////////////////////////////////////
	/// Values getters

	/// Returns value of a constant property. Returns NaN if such property doesn't exist.
	double GetConstPropertyValue(unsigned _nType) const;
	/// Returns value of a TP-property by specified temperature [K] and pressure [Pa]. Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetTPPropertyValue(unsigned _nType, double _dT, double _dP) const;
	/// Returns value of a TP-property by last used temperature and pressure (or for normal conditions for the first call) or value of a constant property.
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetPropertyValue(unsigned _nType) const;

	//////////////////////////////////////////////////////////////////////////
	/// Values setters

	/// Sets value of a constant property.
	void SetConstPropertyValue(unsigned _nPropType, double _dValue);
	/// Sets correlation of a TP-property by specified temperature and pressure interval.
	void SetTPPropertyCorrelation(unsigned _nPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams);
	/// Sets value of a constant property or constant correlation of a TP-property for an entire temperature and pressure interval.
	void SetPropertyValue(unsigned _nPropType, double _dValue);

	//////////////////////////////////////////////////////////////////////////
	/// Other functions

	/// Initializes compound and TPProperties with specified static conditions _dT and _dP. Then GetPropertyValue(Type)/GetValue(void) functions can be called instead
	/// of GetTPPropertyValue(Type,T,P)/GetValue(T,P) for this compound and TPProperties to return data for specified _dT and _dP. Calling GetTPPropertyValue(Type,newT,newP)/GetValue(newT,newP)
	/// after that will reinitialize this compound or TPProperty with newT and newP for all next calls of GetPropertyValue(Type)/GetValue(void).
	void InitializeConditions(double _dT, double _dP);

	//////////////////////////////////////////////////////////////////////////
	/// Save/Load

	/// Saves Compound to protobuf file.
	void SaveToProtobuf(ProtoCompound& _protoCompound);
	/// Loads Compound from protobuf file.
	void LoadFromProtobuf(const ProtoCompound& _protoCompound);

private:
	std::string m_sCompoundName;
	std::string m_sUniqueKey;
	std::string m_sAuthorName;
	SDate m_CreationDate;
	CColor m_Color;

	unsigned m_nPhaseState;
	std::vector<CTPProperty*> m_vProperties;
	std::vector<CConstProperty*> m_vConstProperties;
};