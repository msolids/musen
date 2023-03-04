/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "Bond.h"

class CLiquidBond : public CBond
{
public:
	double m_dSurfaceTension; // [N/m]
	double m_dContactAngle;

	CVector3 m_vecResultForce; // this is force due to change of the bond in the bond coordinate system
	CVector3 m_ResultUnsymMoment;


	CLiquidBond(unsigned _id, CDemStorage* _storage);
	inline double GetSurfaceTension() const { return m_dSurfaceTension; }

	inline void SetViscosity(const double& _dViscosity) { m_dViscosity = _dViscosity; UpdatePrecalculatedValues(); }
	inline void SetContactAngle(const double& _dNewAngle) { m_dContactAngle = _dNewAngle; }
	void UpdateCompoundProperties(const CCompound* _pCompound) override;

private:
	void UpdatePrecalculatedValues() override;
};
