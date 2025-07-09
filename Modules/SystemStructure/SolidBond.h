/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "Bond.h"

class CSolidBond : public CBond
{
public:
	double m_dYoungModulus;
	double m_dPoissonRatio;
	double m_dShearModulus;
	double m_dNormalStrength;
	double m_dTangentialStrength;
	double m_dTimeThermExpCoeff;
	double m_dYieldStrength;
	double m_thermalConductivity{};

	double m_dAxialMoment;		//!!! Should be calculated   //[I]
	double m_dCrossCutSurface;	// The cross cut surface of the bond [m2]

	CVector3 m_NormalForce;
	CVector3 m_TangentialForce;
	CVector3 m_NormalMoment;
	CVector3 m_TangentialMoment;
	CVector3 m_TangDisplacement;
public:
	CSolidBond(unsigned _id, CDemStorage* _storage);

	/// Clones the data from another object of the same type.
	void CloneData(const CPhysicalObject& _other) override;

	//////////////////////////////////////////////////////////////////////////
	/// Sequential getters of time-dependent data.
	CVector3 GetOldTangentialOverlap(double _time) const { return GetAngleAcceleration(_time); } // is used only for old file transformation
	CVector3 GetTangentialOverlap(double _time) const { return GetAngleVelocity(_time); }

	//////////////////////////////////////////////////////////////////////////
	/// Parallel getters of time-dependent data.
	CVector3 GetOldTangentialOverlap() const { return GetAngleAcceleration(); } // is used only for old file transformation
	CVector3 GetTangentialOverlap() const { return GetAngleVelocity(); }

	//////////////////////////////////////////////////////////////////////////
	/// Sequential setters of time-dependent data.
	void SetTangentialOverlap(double _time, const CVector3& _tangOverlap) const { SetAngleVelocity(_time, _tangOverlap); } // Save tangential overlap as angle velocity field.

	//////////////////////////////////////////////////////////////////////////
	/// Parallel setters of time-dependent data. To set the time point for parallel access, call CSystemStructure::PrepareTimePointForWrite(time).
	void SetTangentialOverlap(const CVector3& _tangOverlap) const { SetAngleVelocity(_tangOverlap); } // Save tangential overlap as angle velocity field.

	inline double GetNormalStrength() const { return m_dNormalStrength; };
	inline double GetTangStrnegth() const { return m_dTangentialStrength; };
	inline double GetYoungModulus() const { return m_dYoungModulus; };
	inline double GetYieldStrength() const { return m_dYieldStrength; };
	inline double GetShearModulus() const { return m_dShearModulus; };
	inline double GetTimeThermExpCoeff() const { return m_dTimeThermExpCoeff; }
	double GetThermalConductivity() const { return m_thermalConductivity; }

	void UpdateCompoundProperties(const CCompound* _pCompound) override;

private:
	void UpdatePrecalculatedValues() override;
};
