/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <vector>
#include <cmath>
#include "DefinesMDB.h"

class CCorrelation
{
private:
	double m_dT1;						// Temperature in Kelvin. Left boundary of correlation interval.
	double m_dT2;						// Temperature in Kelvin. Right boundary of correlation interval.
	double m_dP1;						// Pressure in Pascal. Left boundary of correlation interval.
	double m_dP2;						// Pressure in Pascal. Right boundary of correlation interval.
	ECorrelationTypes m_nCorType;		// Type of this correlation.
	std::vector<double> m_vParameters;	// Correlation parameters.

public:
	CCorrelation();
	CCorrelation(double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams);

	/// Returns start T.
	double GetT1() const;
	/// Sets start T.
	void SetT1(double _dT);
	/// Returns end T.
	double GetT2() const;
	/// Sets end T.
	void SetT2(double _dT);
	/// Returns start P	.
	double GetP1() const;
	/// Sets start P.
	void SetP1(double _dP);
	/// Returns end P.
	double GetP2() const;
	/// Sets end P.
	void SetP2(double _dP);
	/// Returns type of the correlation.
	ECorrelationTypes GetType() const;
	/// Sets correlation type.
	void SetType(ECorrelationTypes _nType);
	/// Returns vector of parameters.
	std::vector<double> GetParameters() const;
	/// Sets correlation parameters.
	bool SetParameters(const std::vector<double>& _vParams);

	/// Sets parameters for the correlation. Returns 'true' if correlation has been successfully set.
	bool SetCorrelation(double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nType, const std::vector<double>& _vParams);

	/// Returns value of the correlation. Returns NaN if the correlation type is undefined or correlation is not defined for specified interval.
	double GetCorrelationValue(double _dT, double _dP) const;

	/// Returns true if T lays in specified interval for this correlation.
	bool IsTInInterval(double _dT) const;
	/// Returns true if P lays in specified interval for this correlation.
	bool IsPInInterval(double _dP) const;
	/// Returns true if both parameters are in specified interval for this correlation.
	bool IsInInterval(double _dT, double _dP) const;

	//////////////////////////////////////////////////////////////////////////
	/// Save/Load

	/// Saves Correlation to protobuf file.
	void SaveToProtobuf(ProtoCorrelation& _protoCorrelation);
	/// Loads Correlation from protobuf file.
	void LoadFromProtobuf(const ProtoCorrelation& _protoCorrelation);
};

