/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelSBAerogel.h"

CModelSBAerogel::CModelSBAerogel()
{
	m_name = "Aerogel bond";
	m_uniqueKey = "4CE31CB89C784476AC1971DFC5C39C12";

	AddParameter("PLASTIC_STRAIN_COMPR", "Limited plastic strain for compression [-]", 0.1);
	AddParameter("PLASTIC_STRAIN_TENS", "Limited plastic strain for tension [-]", 0.05);
	AddParameter("SOFTNESS_RATIO_COMPR", "SOFTNESS_RATIO_COMPR", -2.0);
	AddParameter("SOFTNESS_RATIO_TENS", "SOFTNESS_RATIO_TENS", 0.0);
	AddParameter("BREAK_STRAIN_TENS", "BREAK_STRAIN_TENS", 0.5);
	AddParameter("HARDNESS_RATIO_COMPR", "HARDNESS_RATIO_COMPR", 0.1);
	AddParameter("HARDNESS_RATIO_TENS" , "HARDNESS_RATIO_TENS", 0.1);
	m_hasGPUSupport = true;
}

void CModelSBAerogel::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum)   const
{
	// relative angle velocity of contact partners
	CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);

	// the bond in the global coordinate system
	CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	double dDistanceBetweenCenters = currentBond.Length();
	CVector3	rAC = currentBond * 0.5;

	// !! do not delete source equation
	//relativeVelocity = (m_ParticleVelocities[ nLeftParticleID ]-m_ParticleAnglVel[ nLeftParticleID ]*rAC)
	//	- ( m_ParticleVelocities[ nRightParticleID ] + m_ParticleAnglVel[ nRightParticleID ]*rAC);

	// optimized
	CVector3 sumAngleVelocity = Particles().AnglVel(_iLeft) + Particles().AnglVel(_iRight);
	CVector3 relativeVelocity = Particles().Vel(_iLeft) - Particles().Vel(_iRight) - sumAngleVelocity * rAC;

	CVector3 currentContact = currentBond / dDistanceBetweenCenters;
	CVector3 tempVector = _bonds.PrevBond(_iBond)*currentBond;

	CVector3 Phi = currentContact * (DotProduct(sumAngleVelocity, currentContact)*_timeStep*0.5);

	CMatrix3 M(1 + tempVector.z*Phi.z + tempVector.y*Phi.y, Phi.z - tempVector.z - tempVector.y*Phi.x, -Phi.y - tempVector.z*Phi.x + tempVector.y,
		tempVector.z - Phi.z - tempVector.x*Phi.y, tempVector.z*Phi.z + 1 + tempVector.x*Phi.x, -tempVector.z*Phi.y + Phi.x - tempVector.x,
		-tempVector.y - tempVector.x*Phi.z + Phi.y, -tempVector.y*Phi.z + tempVector.x - Phi.x, tempVector.y*Phi.y + tempVector.x*Phi.x + 1);

	CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
	CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

	// normal angle velocity
	CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);
	CVector3 tangAngleVel = relAngleVel - normalAngleVel;

	// calculate the force
	double dPureStrain = (dDistanceBetweenCenters - _bonds.InitialLength(_iBond)) / _bonds.InitialLength(_iBond);
	//double dStrainTotal = dPureStrain - _bonds.NormalPlasticStrain(bond_ID);

	double &dBroken = _bonds.TangentialPlasticStrain(_iBond).x;
	double dPlasticStrainCompr = -fabs(m_parameters[0].value);
	double dPlasticStrainTens  =  fabs(m_parameters[1].value);
	double dSoftnessRatioCompr = m_parameters[2].value;
	double dSoftnessRatioTens  = m_parameters[3].value;
	double dHardnessRatioCompr = m_parameters[5].value;
	double dHardnessRatioTens  = m_parameters[6].value;
	double dBreakageStrainTens = fabs(m_parameters[4].value);
	double dKA = _bonds.CrossCut(_iBond)*_bonds.NormalStiffness(_iBond);

	double dNormalActingForce = 0.;



	if (fabs(dBroken) < 0.5) {
		if ((dPureStrain < 0) && (dPureStrain< dPlasticStrainCompr)){
			double dDeltaPlasticStrain = (dPureStrain - dPlasticStrainCompr);
			dNormalActingForce = dPlasticStrainCompr*dKA + dDeltaPlasticStrain * dSoftnessRatioCompr*dKA;
			_bonds.NormalPlasticStrain(_iBond) = dDeltaPlasticStrain * (1 - dSoftnessRatioCompr);
			if (dNormalActingForce > 0){
				dBroken = -1.;
				dNormalActingForce = 0.;
			}
		} else if ((dPureStrain > 0) && (dPureStrain > dPlasticStrainTens)) {
			double dDeltaPlasticStrain = (dPureStrain - dPlasticStrainTens);
			dNormalActingForce = dPlasticStrainTens * dKA + dDeltaPlasticStrain * dSoftnessRatioTens*dKA;
			_bonds.NormalPlasticStrain(_iBond) = dDeltaPlasticStrain * (1 - dSoftnessRatioTens);
			if (dNormalActingForce < 0){
				dBroken = 1.;
				dNormalActingForce = 0.;
			}
		} else
			dNormalActingForce = (dPureStrain - _bonds.NormalPlasticStrain(_iBond)) * dKA;
		if (dNormalActingForce > (dBreakageStrainTens*dKA))// tension
		{
			_bonds.Active(_iBond) = false;
			_bonds.EndActivity(_iBond) = _time;
			*_pBrokenBondsNum += 1;
		}
	}
	else if (dBroken < -0.5) {
		if (dPureStrain < _bonds.NormalPlasticStrain(_iBond)) {
			_bonds.NormalPlasticStrain(_iBond) = dPureStrain;
		}
		dNormalActingForce = dKA * dHardnessRatioCompr * (dPureStrain - _bonds.NormalPlasticStrain(_iBond));
	}
	else if (dBroken > 0.5) {
		if (dPureStrain > _bonds.NormalPlasticStrain(_iBond)) {
			_bonds.NormalPlasticStrain(_iBond) = dPureStrain;
		}
		dNormalActingForce = dKA * dHardnessRatioTens * (dPureStrain - _bonds.NormalPlasticStrain(_iBond));
	}

	CVector3 vNormalForce = -1 * currentContact * dNormalActingForce;
	_bonds.TangentialOverlap(_iBond) = M * _bonds.TangentialOverlap(_iBond) - tangentialVelocity * _timeStep;
	_bonds.TangentialForce(_iBond) = _bonds.TangentialOverlap(_iBond)*(_bonds.TangentialStiffness(_iBond)*_bonds.CrossCut(_iBond) / _bonds.InitialLength(_iBond));
	_bonds.NormalMoment(_iBond) = M * _bonds.NormalMoment(_iBond) - normalAngleVel * (_timeStep* 2 * _bonds.AxialMoment(_iBond)*_bonds.TangentialStiffness(_iBond) / _bonds.InitialLength(_iBond));
	_bonds.TangentialMoment(_iBond) = M * _bonds.TangentialMoment(_iBond) - tangAngleVel * (_timeStep*_bonds.NormalStiffness(_iBond)*_bonds.AxialMoment(_iBond) / _bonds.InitialLength(_iBond));
	_bonds.TotalForce(_iBond) = vNormalForce + _bonds.TangentialForce(_iBond);

	_bonds.UnsymMoment(_iBond) = rAC * _bonds.TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;
}

void CModelSBAerogel::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
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
