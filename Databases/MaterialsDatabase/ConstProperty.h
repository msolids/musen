/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BaseProperty.h"

class CConstProperty : public CBaseProperty
{
private:
	double m_dValue;

public:
	CConstProperty();
	CConstProperty(unsigned _nType, const std::string& _sName, const std::string& _sUnits);
	~CConstProperty();

	/// Returns constant value of the property.
	double GetValue() const;
	/// Returns constant value of the property.
	double GetValue(double _dT, double _dP) const;
	/// Sets constant value of the property.
	void SetValue(double _dValue);

	//////////////////////////////////////////////////////////////////////////
	/// Save/Load

	/// Saves ConstProperty to protobuf file.
	void SaveToProtobuf(ProtoConstProperty& _protoConstProperty);
	/// Loads ConstProperty from protobuf file.
	void LoadFromProtobuf(const ProtoConstProperty& _protoConstProperty);
};

