/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBKelvin.h"

CModelSBKelvin::CModelSBKelvin()
{
	m_name = "Kelvin bond";
	m_uniqueKey = "09405654485642A686561E1FC646DF1E";
	m_helpFileName = "/Solid Bond/Kelvin.pdf";

	AddParameter("CONSIDER_BREAKAGE", "Consider breakage Yes=1/No=0", 1);
	AddParameter("MU", "MU", 1);
	m_hasGPUSupport = true;
}

void CModelSBKelvin::CalculateSBForce(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrockenBondsNum)   const
{
	// relative angle velocity of contact partners
	CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);

	double dMu = m_parameters[1].value;
	// the bond in the global coordinate system
	CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	double dDistanceBetweenCenters = currentBond.Length();
	CVector3	rAC = currentBond*0.5;

	// !! do not delete source equation
	//relativeVelocity = (m_ParticleVelocities[ nLeftParticleID ]-m_ParticleAnglVel[ nLeftParticleID ]*rAC)
	//	- ( m_ParticleVelocities[ nRightParticleID ] + m_ParticleAnglVel[ nRightParticleID ]*rAC);

	// optimized
	CVector3 sumAngleVelocity = Particles().AnglVel(_iLeft) + Particles().AnglVel(_iRight);
	CVector3 relativeVelocity = Particles().Vel(_iLeft) - Particles().Vel(_iRight) - sumAngleVelocity*rAC;

	CVector3 currentContact = currentBond/dDistanceBetweenCenters;
	CVector3 tempVector = Bonds().PrevBond(_iBond)*currentBond;

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
	double dStrainTotal = (dDistanceBetweenCenters-Bonds().InitialLength(_iBond)) / Bonds().InitialLength(_iBond);

	CVector3 vNormalForce = currentContact*(-1*Bonds().CrossCut(_iBond)*Bonds().NormalStiffness(_iBond)*dStrainTotal);

	CVector3 vDampingForce = -dMu *normalVelocity*Bonds().CrossCut(_iBond)*Bonds().NormalStiffness(_iBond)*fabs(dStrainTotal);
	if (vDampingForce.Length() > 0.5* vNormalForce.Length())
		vDampingForce *= 0.5*vNormalForce.Length() / vDampingForce.Length();

	_bonds.TangentialOverlap(_iBond) = M* Bonds().TangentialOverlap(_iBond) - tangentialVelocity*_timeStep;
	_bonds.TangentialForce(_iBond) = Bonds().TangentialOverlap(_iBond)*(Bonds().TangentialStiffness(_iBond)*Bonds().CrossCut(_iBond) / Bonds().InitialLength(_iBond));

	CVector3 vDampingTangForce = -dMu * tangentialVelocity*Bonds().TangentialOverlap(_iBond).Length()*(Bonds().TangentialStiffness(_iBond)*Bonds().CrossCut(_iBond) / Bonds().InitialLength(_iBond));
	if (vDampingTangForce.Length() > 0.5*Bonds().TangentialForce(_iBond).Length())
		vDampingTangForce *= 0.5*Bonds().TangentialForce(_iBond).Length() / vDampingTangForce.Length();

	_bonds.NormalMoment(_iBond) = M* Bonds().NormalMoment(_iBond)- normalAngleVel*(_timeStep* 2 * Bonds().AxialMoment(_iBond)*Bonds().TangentialStiffness(_iBond)/ Bonds().InitialLength(_iBond));
	_bonds.TangentialMoment(_iBond) = M* Bonds().TangentialMoment(_iBond) - tangAngleVel*(_timeStep*Bonds().NormalStiffness(_iBond)*Bonds().AxialMoment(_iBond)/ Bonds().InitialLength(_iBond));
	_bonds.TotalForce(_iBond) = vNormalForce + Bonds().TangentialForce(_iBond) + vDampingForce + vDampingTangForce;

	_bonds.UnsymMoment(_iBond) = currentBond * 0.5*Bonds().TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;

	if (m_parameters[0].value == 0 ) return; // consider breakage
	// check the bond destruction
	double dForceLength = vNormalForce.Length();
	if (dStrainTotal <= 0)	// compression
		dForceLength *= -1;
	double dMaxStress = dForceLength / Bonds().CrossCut(_iBond) + Bonds().TangentialMoment(_iBond).Length()*Bonds().Diameter(_iBond) / (2 * Bonds().AxialMoment(_iBond));
	double dMaxTorque = Bonds().TangentialForce(_iBond).Length() / Bonds().CrossCut(_iBond) + Bonds().NormalMoment(_iBond).Length()*Bonds().Diameter(_iBond) / (2 * 2 * Bonds().AxialMoment(_iBond));

	if ( ( dMaxStress  >= Bonds().NormalStrength(_iBond) ) || ( dMaxTorque >= Bonds().TangentialStrength(_iBond)) )
	{
		_bonds.Active(_iBond) = false;
		_bonds.EndActivity(_iBond) = _time;
		*_pBrockenBondsNum += 1;
	}
}

