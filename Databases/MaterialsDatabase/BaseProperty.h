/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "DefinesMDB.h"
#include <string>

class CBaseProperty
{
protected:
	std::string m_sName;		// Name of the property.
	std::string m_sUnits;		// Property units [Pa, s, ...].
	unsigned m_nPropertyType;	// Type of the property (density, heat capacity, viscosity, ... ).

public:
	CBaseProperty();
	CBaseProperty(unsigned _nType, const std::string& _sName, const std::string& _sUnits);
	virtual ~CBaseProperty() = 0;

	std::string GetName() const;
	void SetName(const std::string& _sName);

	std::string GetUnits() const;
	void SetUnits(const std::string& _sUnits);

	unsigned GetType() const;
	void SetType(unsigned _nType);

	virtual double GetValue() const;
	virtual double GetValue(double _dT, double _dP) const;
};

