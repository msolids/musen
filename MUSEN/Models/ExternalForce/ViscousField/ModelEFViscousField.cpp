/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFViscousField.h"

CModelEFViscousField::CModelEFViscousField()
{
	m_name = "Viscous Field";
	m_uniqueKey = "1C8F05E0516349A78BF1359006D61003";
	m_helpFileName = "/External Force/ViscousField.pdf";

	AddParameter("MEDIUM_VISCOSITY", "Medium viscosity [m2/s]", 1.34*1e-5);
	AddParameter("MEDIUM_DENSITY", "Density of medium [kg/m3]", 1.28);

	m_hasGPUSupport = true;
}

void CModelEFViscousField::CalculateEFForce(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const
{
	double dKinViscosity = m_parameters[ 0 ].value;
	double dMediumDensity = m_parameters[ 1 ].value;
	////double dInitVel = 2;
	//double dPosition = _pPart.vCoord.Length();
	//CVector3 vFlowVel = sin( _dTime*10 )*dInitVel/(dPosition*dPosition)*_pPart.vCoord.Normalized();
	//if ( dPosition < 0.1 )
		//vFlowVel = vFlowVel*0;
	CVector3 vRelVel = Particles().Vel(_iPart);
	double dRelVelLength = vRelVel.Length();
	double dReynolds = Particles().Radius(_iPart) *2*dRelVelLength/dKinViscosity;
	if ( dReynolds == 0 ) return; // no drag
	double dCd;
	if ( dReynolds < 0.5 )
		dCd = 24.0/dReynolds;
	else if ( dReynolds < 10.1 )
		dCd = 27.0/pow( dReynolds, 0.8 );
	else
		dCd = 17.0/pow( dReynolds, 0.6 );

	_particles.Force(_iPart) -= vRelVel.Normalized()*dCd*PI*pow(Particles().Radius(_iPart), 2 )*dMediumDensity/2*pow( dRelVelLength, 2 );
}
