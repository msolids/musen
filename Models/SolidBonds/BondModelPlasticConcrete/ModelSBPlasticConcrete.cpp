/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBPlasticConcrete.h"

CModelSBPlasticConcrete::CModelSBPlasticConcrete()
{
	m_name = "Plastic concrete bond";
	m_uniqueKey = "CA3CE500F3FE4E37914SS7D544553847";

	AddParameter("CONSIDER_BREAKAGE", "Consider breakage Yes=1/No=0", 1);
	//AddParameter("COMPRESSIVE_BREAK", "Consider compressive breakage Yes=1/No=0", 0);

	m_hasGPUSupport = true;
}

void CModelSBPlasticConcrete::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum)   const
{
	// the bond in the global coordinate system
	CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	double dDistanceBetweenCenters = currentBond.Length();
	CVector3 rAC = currentBond * 0.5;

	double dDruckYieldStrength = -_bonds.YieldStrength(_iBond);
	double dZugYieldStrength = 0.7*_bonds.YieldStrength(_iBond);
	double dBetta = 0.5;
	double dAlpha = 0.3;
	double dKn = _bonds.NormalStiffness(_iBond);

	// relative angle velocity of contact partners
	CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);

	// optimized
	CVector3 sumAngleVelocity = Particles().AnglVel(_iLeft) + Particles().AnglVel(_iRight);
	CVector3 relativeVelocity = Particles().Vel(_iLeft) - Particles().Vel(_iRight) - sumAngleVelocity*rAC;

	CVector3 currentContact = currentBond/dDistanceBetweenCenters;
	CVector3 tempVector = _bonds.PrevBond(_iBond)*currentBond;
	CVector3 Phi = currentContact*(DotProduct(sumAngleVelocity, currentContact)*_timeStep*0.5);

	CMatrix3 M( 1 + tempVector.z*Phi.z + tempVector.y*Phi.y,	Phi.z - tempVector.z - tempVector.y*Phi.x,		-Phi.y - tempVector.z*Phi.x + tempVector.y,
		tempVector.z - Phi.z - tempVector.x*Phi.y,		tempVector.z*Phi.z + 1 + tempVector.x*Phi.x,	-tempVector.z*Phi.y + Phi.x - tempVector.x,
		-tempVector.y - tempVector.x*Phi.z + Phi.y,		-tempVector.y*Phi.z + tempVector.x - Phi.x,		tempVector.y*Phi.y + tempVector.x*Phi.x + 1);

	CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
	CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

	// normal angle velocity
	CVector3 normalAngleVel = currentContact*DotProduct(currentContact, relAngleVel);
	CVector3 tangAngleVel = relAngleVel - normalAngleVel;

	// calculate the force
	double dStrainTotal = (dDistanceBetweenCenters-_bonds.InitialLength(_iBond)) / _bonds.InitialLength(_iBond);
	double dElasticStrain = dStrainTotal - _bonds.NormalPlasticStrain(_iBond);
	double dCurrentNormalStress;
	if (dElasticStrain >= 0) //tension
	{
		dCurrentNormalStress = dElasticStrain * _bonds.NormalStiffness(_iBond);
		double dLimitStress = dZugYieldStrength - _bonds.NormalPlasticStrain(_iBond)*dKn*dAlpha;
		if (dCurrentNormalStress > dLimitStress)
		{
			_bonds.NormalPlasticStrain(_iBond) += (dCurrentNormalStress - dLimitStress)*(1 + dAlpha ) / dKn;
			dElasticStrain = dStrainTotal - _bonds.NormalPlasticStrain(_iBond);
			dCurrentNormalStress = dElasticStrain * _bonds.NormalStiffness(_iBond);
		}
		if (m_parameters[0].value != 0.0 && dCurrentNormalStress < 0) // bond breakage
		{
			_bonds.Active(_iBond) = false;
			_bonds.EndActivity(_iBond) = _time;
			*_pBrokenBondsNum += 1;
		}
	}
	else // druck
	{
		dCurrentNormalStress = dElasticStrain * _bonds.NormalStiffness(_iBond);
		double dLimitStress = dDruckYieldStrength+ _bonds.NormalPlasticStrain(_iBond)*dKn*dBetta;
		if ( dCurrentNormalStress < dLimitStress)
		{
			_bonds.NormalPlasticStrain(_iBond) += (dCurrentNormalStress - dLimitStress)*(1 - dBetta) / dKn;
			dElasticStrain = dStrainTotal - _bonds.NormalPlasticStrain(_iBond);
			dCurrentNormalStress = dElasticStrain * _bonds.NormalStiffness(_iBond);
		}
	}
	CVector3 vNormalForce = currentContact* dCurrentNormalStress*_bonds.CrossCut(_iBond);

	_bonds.TangentialOverlap(_iBond) = M*_bonds.TangentialOverlap(_iBond) - tangentialVelocity*_timeStep;
	_bonds.TangentialPlasticStrain(_iBond) = M*_bonds.TangentialPlasticStrain(_iBond);
	CVector3 vTangStrain = _bonds.TangentialOverlap(_iBond) / _bonds.InitialLength(_iBond);
	CVector3 vTangStress = (_bonds.TangentialOverlap(_iBond) - _bonds.TangentialPlasticStrain(_iBond))*_bonds.TangentialStiffness(_iBond);

	_bonds.TangentialForce(_iBond) = vTangStress*_bonds.CrossCut(_iBond);
	_bonds.NormalMoment(_iBond) = M*_bonds.NormalMoment(_iBond) - normalAngleVel*(_timeStep* 2 * _bonds.AxialMoment(_iBond)*_bonds.TangentialStiffness(_iBond)) / _bonds.InitialLength(_iBond);
	_bonds.TangentialMoment(_iBond) = M*_bonds.TangentialMoment(_iBond) - tangAngleVel*(_timeStep*_bonds.NormalStiffness(_iBond)*_bonds.AxialMoment(_iBond)) / _bonds.InitialLength(_iBond);
	_bonds.TotalForce(_iBond) = vNormalForce + _bonds.TangentialForce(_iBond);

	_bonds.UnsymMoment(_iBond) = rAC*_bonds.TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;
}

void CModelSBPlasticConcrete::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
{
	if (Bonds().LeftID(_iBond) == _iPart)
	{
		_particles.Force(_iPart) += Bonds().TotalForce(_iBond);
		_particles.Moment(_iPart) += Bonds().NormalMoment(_iBond) + Bonds().TangentialMoment(_iBond) - Bonds().UnsymMoment(_iBond);
	}
	else if (Bonds().RightID(_iBond) == _iPart)
	{
		_particles.Force(_iPart) -= Bonds().TotalForce(_iBond);
		_particles.Moment(_iPart) -= Bonds().NormalMoment(_iBond) + Bonds().TangentialMoment(_iBond) + Bonds().UnsymMoment(_iBond);
	}
}
