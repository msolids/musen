/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "TPProperty.h"

class ProtoInteraction;

class CInteraction
{
private:
	std::string m_sCompoundKey1;				/// Key to identify first compound.
	std::string m_sCompoundKey2;				/// Key to identify second compound.
	std::vector<CTPProperty*> m_vProperties;	/// List of interaction properties

public:
	CInteraction(const std::string& _sKey1, const std::string& _sKey2);
	CInteraction(const CInteraction& _interaction);
	~CInteraction();

	/// Copies data from other interaction.
	CInteraction& operator=(const CInteraction& _interaction);

	/// Removes all properties from interaction.
	void Clear();
	/// Copies data from another interaction.
	void Copy(const CInteraction& _interaction);

	/// Returns key of the first compound.
	std::string GetKey1() const;
	/// Returns key of the second compound.
	std::string GetKey2() const;
	/// Sets keys of compounds.
	void SetKeys(const std::string& _sKey1, const std::string& _sKey2);
	/// Returns number of properties.
	size_t PropertiesNumber() const;

	//////////////////////////////////////////////////////////////////////////
	/// Pointers getters

	/// Returns pointer to a specified TP property. Returns nullptr if property is not found.
	CTPProperty* GetProperty(unsigned _nType);
	/// Returns constant pointer to a specified TP property. Returns nullptr if property is not found.
	const CTPProperty* GetProperty(unsigned _nType) const;
	/// Returns pointer to a specified TP property by its index. Returns nullptr if property is not found.
	CTPProperty* GetPropertyByIndex(size_t _nIndex);
	/// Returns constant pointer to a specified TP property by its index. Returns nullptr if property is not found.
	const CTPProperty* GetPropertyByIndex(size_t _nIndex) const;

	//////////////////////////////////////////////////////////////////////////
	/// Values getters

	/// Returns value of a TP-property by specified temperature [K] and pressure [Pa]. Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetTPPropertyValue(unsigned _nType, double _dT, double _dP) const;
	/// Returns value of a TP-property by last used temperature and pressure (or for normal conditions for the first call).
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetPropertyValue(unsigned _nType) const;

	//////////////////////////////////////////////////////////////////////////
	/// Values setters

	/// Sets correlation of a TP-property by specified temperature and pressure interval.
	void SetTPPropertyCorrelation(unsigned _nPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams);
	/// Sets value of a constant correlation of a TP-property for an entire temperature and pressure interval.
	void SetPropertyValue(unsigned _nPropType, double _dValue);

	//////////////////////////////////////////////////////////////////////////
	/// Other functions

	/// Initializes Interaction with specified static conditions _dT and _dP. Then GetPropertyValue(Type) function can be called instead GetTPPropertyValue(Type,T,P) for this Interaction to return
	/// data for specified _dT and _dP. Calling GetTPPropertyValue(Type,newT,newP) after that will reinitialize this Interaction with newT and newP for all next calls of GetPropertyValue(Type).
	void InitializeConditions(double _dT, double _dP);

	/// Returns true if this interaction describes relation between specified compounds.
	bool IsBetween(const std::string& _sKey1, const std::string& _sKey2) const;

	//////////////////////////////////////////////////////////////////////////
	/// Save/Load

	/// Saves Mixture to protobuf-file
	void SaveToProtobuf(ProtoInteraction& _protoInteraction);
	/// Loads Mixture from protobuf-file
	void LoadFromProtobuf(const ProtoInteraction& _protoInteraction);
};

