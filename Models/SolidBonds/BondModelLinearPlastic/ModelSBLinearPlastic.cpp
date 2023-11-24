/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBLinearPlastic.h"

CModelSBLinearPlastic::CModelSBLinearPlastic()
{
	m_name = "Linear plastic bond";
	m_uniqueKey = "4CE31CB89C784476AC1971DFC5C39C10";

	AddParameter("CONSIDER_BREAKAGE", "Consider breakage Yes=1/No=0", 1);
	AddParameter("KLoad", "Loading stiffness", 1e+6);
	AddParameter("KUnLoad", "Unloading stiffness", 1e+6);

	m_hasGPUSupport = true;
}

void CModelSBLinearPlastic::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum)   const
{
	// relative angle velocity of contact partners
	CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);

	// the bond in the global coordinate system
	CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	double dDistanceBetweenCenters = currentBond.Length();
	CVector3	rAC = currentBond*0.5;

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
	double dKload = m_parameters[1].value;
	double dKUnload = m_parameters[2].value;
	CVector3 vNormalForce;

	if (_bonds.NormalStiffness(_iBond) <= 2.0)
	{
		if (fabs(dKUnload*(dStrainTotal - _bonds.NormalPlasticStrain(_iBond))) > fabs(dKload*dStrainTotal)) // loading stage - increase plastic strain
		{
			vNormalForce = currentContact * (-1 * _bonds.CrossCut(_iBond)*dKload*dStrainTotal*_bonds.NormalStiffness(_iBond));
			_bonds.NormalPlasticStrain(_iBond) = dStrainTotal * (1 - dKload / dKUnload);
		}
		else
			vNormalForce = currentContact * (-1 * _bonds.CrossCut(_iBond)*dKUnload*(dStrainTotal - _bonds.NormalPlasticStrain(_iBond))*_bonds.NormalStiffness(_iBond));
	}
	else
		vNormalForce = currentContact * (-1 * _bonds.CrossCut(_iBond)*_bonds.NormalStiffness(_iBond)*dStrainTotal);

	_bonds.TangentialOverlap(_iBond) = M*_bonds.TangentialOverlap(_iBond) - tangentialVelocity*_timeStep;
	_bonds.TangentialForce(_iBond) = _bonds.TangentialOverlap(_iBond)*(_bonds.TangentialStiffness(_iBond)*_bonds.CrossCut(_iBond)/ _bonds.InitialLength(_iBond));
	_bonds.NormalMoment(_iBond) = M*_bonds.NormalMoment(_iBond)- normalAngleVel*(_timeStep* 2 * _bonds.AxialMoment(_iBond)*_bonds.TangentialStiffness(_iBond)/ _bonds.InitialLength(_iBond));
	_bonds.TangentialMoment(_iBond) = M*_bonds.TangentialMoment(_iBond) - tangAngleVel*(_timeStep*_bonds.NormalStiffness(_iBond)*_bonds.AxialMoment(_iBond)/ _bonds.InitialLength(_iBond));
	_bonds.TotalForce(_iBond) = vNormalForce + _bonds.TangentialForce(_iBond);

	_bonds.UnsymMoment(_iBond) = rAC*_bonds.TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;

	if (m_parameters[0].value == 0 ) return; // consider breakage
	// check the bond destruction
	double dForceLength = vNormalForce.Length();
	if (dStrainTotal <= 0)	// compression
		dForceLength *= -1;
	double dMaxStress = dForceLength / _bonds.CrossCut(_iBond) + _bonds.TangentialMoment(_iBond).Length()*_bonds.Diameter(_iBond) / (2 * _bonds.AxialMoment(_iBond));
	double dMaxTorque = _bonds.TangentialForce(_iBond).Length() / _bonds.CrossCut(_iBond) + _bonds.NormalMoment(_iBond).Length()*_bonds.Diameter(_iBond) / (2 * 2 * _bonds.AxialMoment(_iBond));

	if ( ( dMaxStress >= _bonds.NormalStrength(_iBond) ) || ( dMaxTorque >= _bonds.TangentialStrength(_iBond)) )
	{
		_bonds.Active(_iBond) = false;
		_bonds.EndActivity(_iBond) = _time;
		*_pBrokenBondsNum += 1;
	}
}

void CModelSBLinearPlastic::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
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
