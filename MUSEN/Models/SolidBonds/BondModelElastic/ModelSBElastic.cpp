/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBElastic.h"

CModelSBElastic::CModelSBElastic()
{
	m_name = "Elastic bond";
	m_uniqueKey = "4CE31CB89C784476AC1971DFC5C39C10";
	m_helpFileName = "/Solid Bond/Elastic.pdf";

	AddParameter("CONSIDER_BREAKAGE", "Consider breakage Crit_2/Crit_1=1/No=0", 1);

	m_hasGPUSupport = true;
}

void CModelSBElastic::CalculateSBForce(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const
{
	// relative angle velocity of contact partners
	CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);

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

	_bonds.TangentialOverlap(_iBond) = M*Bonds().TangentialOverlap(_iBond) - tangentialVelocity*_timeStep;
	_bonds.TangentialForce(_iBond) = Bonds().TangentialOverlap(_iBond)*(Bonds().TangentialStiffness(_iBond)*Bonds().CrossCut(_iBond)/ Bonds().InitialLength(_iBond));
	_bonds.NormalMoment(_iBond) = M*Bonds().NormalMoment(_iBond)- normalAngleVel*(_timeStep* 2 * Bonds().AxialMoment(_iBond)*Bonds().TangentialStiffness(_iBond)/ Bonds().InitialLength(_iBond));
	_bonds.TangentialMoment(_iBond) = M*Bonds().TangentialMoment(_iBond) - tangAngleVel*(_timeStep*Bonds().NormalStiffness(_iBond)*Bonds().AxialMoment(_iBond)/ Bonds().InitialLength(_iBond));
	_bonds.TotalForce(_iBond) = vNormalForce + Bonds().TangentialForce(_iBond);

	_bonds.UnsymMoment(_iBond) = rAC*Bonds().TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;

	if (m_parameters[0].value == 0.0) return; // consider breakage

	// check the bond destruction
	double forceLength = vNormalForce.Length();
	if ( dStrainTotal <= 0)	// compression
		forceLength *= -1;
	const double maxStress1 = forceLength / Bonds().CrossCut(_iBond);
	const double maxStress2 = Bonds().TangentialMoment(_iBond).Length()*Bonds().Diameter(_iBond) / (2.0 * Bonds().AxialMoment(_iBond));
	const double maxTorque1 = Bonds().TangentialForce(_iBond).Length() / Bonds().CrossCut(_iBond);
	const double maxTorque2 = Bonds().NormalMoment(_iBond).Length()*Bonds().Diameter(_iBond) / (4.0 * _bonds.AxialMoment(_iBond));

	bool bondBreaks = false;
	if (m_parameters[0].value == 1.0 &&		// standard breakage criteria
		  (maxStress1 + maxStress2 >= Bonds().NormalStrength(_iBond)
		|| maxTorque1 + maxTorque2 >= Bonds().TangentialStrength(_iBond)))
		bondBreaks = true;
	if (m_parameters[0].value == 2.0 &&		// alternative breakage criteria
		  (maxStress1 >= Bonds().NormalStrength(_iBond)
		|| maxStress2 >= Bonds().NormalStrength(_iBond) && dStrainTotal > 0
		|| maxTorque1 >= Bonds().TangentialStrength(_iBond)
		|| maxTorque2 >= Bonds().TangentialStrength(_iBond)))
		bondBreaks = true;
	if (bondBreaks)
	{
		_bonds.Active(_iBond) = false;
		_bonds.EndActivity(_iBond) = _time;
		*_pBrokenBondsNum += 1;
	}
}
