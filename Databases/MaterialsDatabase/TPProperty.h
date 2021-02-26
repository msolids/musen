/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BaseProperty.h"
#include "Correlation.h"
#include <limits>
#include <memory>

class ProtoTPProperty;

class CTPProperty : public CBaseProperty
{
private:
	std::vector<std::unique_ptr<CCorrelation>> m_vCorrelations;	/// List of correlations for different combinations of T and P.
	mutable int m_nLastUsedCorrelation{ -1 };					/// Index of a last accessed correlation or '-1' if no correlations have been specified.
	mutable double m_dLastReturnedValue{ _NOT_A_NUMBER };		/// Stores last returned value of the property.

public:
	CTPProperty();
	CTPProperty(unsigned _nType, const std::string& _sName, const std::string& _sUnits);
	CTPProperty(const CTPProperty& _property);
	~CTPProperty();

	/// Copies data from other compound.
	CTPProperty& operator=(const CTPProperty& _property);

	/// Removes all correlations.
	void Clear();
	/// Copies data from another property.
	void Copy(const CTPProperty& _property);

	//////////////////////////////////////////////////////////////////////////
	/// Values getters

	/// Returns property value. Returns NaN if correlation is not defined for specified interval. Returns NaN if correlation is not defined for specified T and P.
	double GetValue(double _dT, double _dP) const;
	/// Returns last returned property value. For the first call returns value at normal conditions. Returns NaN if correlation is not defined for specified T and P.
	double GetValue() const;

	//////////////////////////////////////////////////////////////////////////
	/// Work with correlations

	/// Returns number of defined correlations.
	size_t CorrelationsNumber() const;
	/// Returns correlation type of the specified correlation or COR_UNDEFINED if such correlation has not been defined.
	ECorrelationTypes GetCorrelationType(size_t _nIndex) const;

	/// Returns copy to a correlation with specified index. Returns empty correlation if such correlation is not defined.
	CCorrelation GetCorrelation(size_t _nIndex) const;
	/// Returns copy to a correlation for specified interval. Returns empty correlation if nothing is defined for this region.
	CCorrelation GetCorrelation(double _dT, double _dP) const;


	/// Adds constant correlation to the property with value DEFAULT_CORR_VALUE, returns index of the correlation.
	size_t AddCorrelation();
	/// Adds constant correlation to the property with value _dConstValue, returns index of the correlation.
	size_t AddCorrelation(double _dConstValue);
	/// Adds specified correlation to the property, returns index of the correlation.
	size_t AddCorrelation(double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams);
	/// Sets correlation settings.
	void SetCorrelation(size_t _nIndex, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams);
	/// Sets const correlation and removes another.
	void SetCorrelation(double _dConstValue);
	/// Removes correlation from the list.
	void RemoveCorrelation(size_t _nIndex);
	/// Removes all defined correlations.
	void RemoveAllCorrelations();

	//////////////////////////////////////////////////////////////////////////
	/// Other functions

	/// Initializes property with specified static conditions _dT and _dP. Then GetValue(void) functions can be called instead of GetValue(T,P) for this property to return data for specified _dT and _dP.
	/// Calling GetValue(newT,newP) after that will reinitialize this property with newT and newP for all next calls of GetValue(void).
	void InitializeConditions(double _dT, double _dP) const;

	//////////////////////////////////////////////////////////////////////////
	/// Save/Load

	/// Saves TPProperty to protobuf file.
	void SaveToProtobuf(ProtoTPProperty& _protoTPProperty);
	/// Loads TPProperty from protobuf file.
	void LoadFromProtobuf(const ProtoTPProperty& _protoTPProperty);

private:
	/// Resets T and P to default normal conditions.
	void ResetConditions() const;
};

