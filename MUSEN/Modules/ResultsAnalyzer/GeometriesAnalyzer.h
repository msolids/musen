/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzer.h"

class CGeometriesAnalyzer : public CResultsAnalyzer
{
	CRealGeometry* m_pObject;	/// The geometry for which everything is calculated.
	CVector3 m_vInitPosition;	/// The geometry's initial position.

	typedef std::function<void(const double& _timePoint)> calcFunction_type;
	std::vector<calcFunction_type> m_vCalcFunctions; // vector with functions calculating the output

public:
	CGeometriesAnalyzer();
	bool Export() override;
	bool InitAnalyzer(std::vector<EPropertyType> _properties);

	void WriteTimePoint(const double& _timePoint);
	void FlushStream() const;

private:
	// Functions to calculate variables
	CVector3 CalculateDistance(const double& _timePoint) const;
	CVector3 CalculateForce(const double& _timePoint) const;
};