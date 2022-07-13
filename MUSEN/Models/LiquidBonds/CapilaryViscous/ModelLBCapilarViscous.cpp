/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelLBCapilarViscous.h"

CModelLBCapilarViscous::CModelLBCapilarViscous()
{
	m_name = "Capilar-Viscous";
	m_uniqueKey = "2A1C321B41ED44138C826EE8322441E5";
	m_helpFileName = "/Liquid Bond/Mikami.pdf";

	AddParameter("MIN_THICKNESS", "Min. thickness of liquid [m]", 1e-5);
	AddParameter("CONTACT_ANGLE", "Contact angle [grad]", 70);
}

void CModelLBCapilarViscous::CalculateLBForce(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SLiquidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const
{
/*	double dMinThickness = m_parameters[ 0 ].value;
	double dContactAngle = m_parameters[ 1 ].value*PI/180;

	CVector3 currentBond = Particles().Coord(_iLeft) - Particles().Coord(_iRight);
	double dCurrentBondLength = currentBond.Length() - Particles().Radius(_iLeft) - Particles().Radius(_iRight);
	if ( dCurrentBondLength < dMinThickness )
		dCurrentBondLength = dMinThickness;
	CVector3 rAC = currentBond*0.5;

	double dEquivRadius = Particles().Radius(_iLeft)*Particles().Radius(_iRight)/(Particles().Radius(_iLeft)+Particles().Radius(_iRight));

	// optimized
	CVector3 sumAngleVelocity = Particles().AnglVel(_iLeft) + Particles().AnglVel(_iRight);
	CVector3 relativeVelocity = Particles().Vel(_iLeft) - Particles().Vel(_iRight) - sumAngleVelocity*rAC;	//	currentContact = currentBond/Length( currentBond );

	CVector3 vNormalVector = currentBond / currentBond.Length();
	CVector3 normalVelocity = vNormalVector * DotProduct(vNormalVector, relativeVelocity);;
	CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

	double dA = -1.1*pow(Bonds().Volume(_iBond), -0.53 );
	double dTempLn = log(Bonds().Volume(_iBond));
	double dB = (-0.34*dTempLn-0.96)*dContactAngle*dContactAngle-0.019*dTempLn+0.48;
	double dC = 0.0042*dTempLn + 0.078;
	CVector3 dCapForce = vNormalVector*(-1)*PI*dEquivRadius*Bonds().SurfaceTension(_iBond)*( exp( dA*dCurrentBondLength + dB ) + dC );
	CVector3 dViscForceNormal = normalVelocity*(-1*6*PI*Bonds().Viscosity(_iBond)*dEquivRadius*dEquivRadius/dCurrentBondLength);

	_bonds.NormalForce(_iBond) = dViscForceNormal + dCapForce;
		//2*PI*dEquivRadius*getBonds().SurfaceTension(bond_ID)*cos( dContactAngle )*(1-pow( 1+2*getBonds().Volume(bond_ID)/(PI*dEquivRadius*dCurrentBondLength*dCurrentBondLength), -0.5 ));
	_bonds.TangentialForce(_iBond) = tangentialVelocity*(-1*6*PI*Bonds().Viscosity(_iBond)*dEquivRadius*(8.0/15.0*log( dEquivRadius/(dCurrentBondLength) ) +0.9588) );
	_bonds.UnsymMoment(_iBond) = rAC* Bonds().TangentialForce(_iBond);

	if ( dCurrentBondLength > (1+0.5*dContactAngle)*pow(Bonds().Volume(_iBond), 1.0/3.0 ) )
	{
		_bonds.Active(_iBond) = false;
		_bonds.EndActivity(_iBond) = _time;
		*_pBrokenBondsNum += 1;
		_bonds.NormalForce(_iBond).Init(0);
		_bonds.TangentialForce(_iBond).Init(0);
	}*/
}