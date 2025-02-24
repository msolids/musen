/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBWeakening.h"
#include "BasicTypes.h"

CModelSBWeakening::CModelSBWeakening()
{
	m_name = "Bond with weakening";
	m_uniqueKey = "CA3CE500F3FE4E33914EC7D544553847";

	/* 0 */ AddParameter("CONSIDER_BREAKAGE", "Consider breakage Yes=1/No=0", 1);
	/* 1 */ AddParameter("COMPRESSIVE_BREAK", "Consider compressive breakage Yes=1/No=0", 0);
	/* 2 */ AddParameter("BREAKAGE_STRAIN", "Breakage strain", 0.1);
	/* 3 */ AddParameter("WEAKENING_FACTOR", "Weakening factor", 0.5);

	m_hasGPUSupport = true;
}

void CModelSBWeakening::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum)   const
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
	double dNormalStress = (dStrainTotal - _bonds.NormalPlasticStrain(_iBond))*_bonds.NormalStiffness(_iBond);

	double dWeakeningFactor = m_parameters[3].value;
	//	if (dStrainTotal > 0 ) // plastic deformation only for tension
	if (fabs(dNormalStress) > _bonds.YieldStrength(_iBond))
	{
		_bonds.NormalPlasticStrain(_iBond) = dStrainTotal - dNormalStress * (dWeakeningFactor*((double)rand()) / RAND_MAX) / _bonds.NormalStiffness(_iBond);
		dNormalStress = (dStrainTotal - _bonds.NormalPlasticStrain(_iBond))*_bonds.NormalStiffness(_iBond);
	}
	CVector3 vNormalForce = currentContact*dNormalStress*(-1*Bonds().CrossCut(_iBond));

	_bonds.TangentialOverlap(_iBond) = M * Bonds().TangentialOverlap(_iBond) - tangentialVelocity * _timeStep;
	double dTangentialStress = Length (Bonds().TangentialOverlap(_iBond)*Bonds().TangentialStiffness(_iBond) / Bonds().InitialLength(_iBond));
	if (dTangentialStress > _bonds.YieldStrength(_iBond))
		_bonds.TangentialOverlap(_iBond) *= dWeakeningFactor*((double)rand()) / RAND_MAX;
	_bonds.TangentialForce(_iBond) = Bonds().TangentialOverlap(_iBond)*(Bonds().TangentialStiffness(_iBond)*Bonds().CrossCut(_iBond) / Bonds().InitialLength(_iBond));
	_bonds.NormalMoment(_iBond) = M * Bonds().NormalMoment(_iBond) - normalAngleVel * (_timeStep * 2 * Bonds().AxialMoment(_iBond)*Bonds().TangentialStiffness(_iBond) / Bonds().InitialLength(_iBond));
	_bonds.TangentialMoment(_iBond) = M * Bonds().TangentialMoment(_iBond) - tangAngleVel * (_timeStep*Bonds().NormalStiffness(_iBond)*Bonds().AxialMoment(_iBond) / Bonds().InitialLength(_iBond));
	_bonds.TotalForce(_iBond) = vNormalForce + Bonds().TangentialForce(_iBond);

	_bonds.UnsymMoment(_iBond) = rAC * Bonds().TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;

	if (m_parameters[0].value == 0 ) return; // consider breakage
	// check the bond destruction

	if (dStrainTotal > m_parameters[2].value || (m_parameters[1].value != 0.0 && dStrainTotal < -2 * m_parameters[2].value)) // strain greater than breakage strain
	{
		_bonds.Active(_iBond) = false;
		_bonds.EndActivity(_iBond) = _time;
		*_pBrokenBondsNum += 1;
	}
}

void CModelSBWeakening::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
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
