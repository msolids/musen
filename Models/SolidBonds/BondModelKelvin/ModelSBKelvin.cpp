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
	m_hasGPUSupport = true;
}

void CModelSBKelvin::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _brokenBondsNum) const
{
	// relative angle velocity of contact partners
	const CVector3 relAngleVel = Particles().AnglVel(_iLeft) - Particles().AnglVel(_iRight);

	const double mu = Bonds().Viscosity(_iBond);
	// the bond in the global coordinate system
	const CVector3 currentBond = GetSolidBond(Particles().Coord(_iRight), Particles().Coord(_iLeft), m_PBC);
	const double distanceBetweenCenters = currentBond.Length();
	const CVector3 rAC = currentBond * 0.5;

	// !! do not delete source equation
	//relativeVelocity = (m_ParticleVelocities[ nLeftParticleID ]-m_ParticleAnglVel[ nLeftParticleID ]*rAC)
	//	- ( m_ParticleVelocities[ nRightParticleID ] + m_ParticleAnglVel[ nRightParticleID ]*rAC);

	// optimized
	const CVector3 sumAngleVelocity = Particles().AnglVel(_iLeft) + Particles().AnglVel(_iRight);
	const CVector3 relativeVelocity = Particles().Vel(_iLeft) - Particles().Vel(_iRight) - sumAngleVelocity * rAC;

	const CVector3 currentContact = currentBond / distanceBetweenCenters;
	const CVector3 tempVector = Bonds().PrevBond(_iBond) * currentBond;

	const CVector3 phi = currentContact * (DotProduct(sumAngleVelocity, currentContact) * _timeStep * 0.5);

	const CMatrix3 M(1 + tempVector.z * phi.z + tempVector.y * phi.y, phi.z - tempVector.z - tempVector.y * phi.x, -phi.y - tempVector.z * phi.x + tempVector.y,
		tempVector.z - phi.z - tempVector.x * phi.y, tempVector.z * phi.z + 1 + tempVector.x * phi.x, -tempVector.z * phi.y + phi.x - tempVector.x,
		-tempVector.y - tempVector.x * phi.z + phi.y, -tempVector.y * phi.z + tempVector.x - phi.x, tempVector.y * phi.y + tempVector.x * phi.x + 1);

	const CVector3 normalVelocity = currentContact * DotProduct(currentContact, relativeVelocity);
	const CVector3 tangentialVelocity = relativeVelocity - normalVelocity;

	// normal angle velocity
	const CVector3 normalAngleVel = currentContact * DotProduct(currentContact, relAngleVel);
	const CVector3 tangAngleVel = relAngleVel - normalAngleVel;

	// calculate the force
	const double strainTotal = (distanceBetweenCenters - Bonds().InitialLength(_iBond)) / Bonds().InitialLength(_iBond);

	const CVector3 normalForce = currentContact * (-1 * Bonds().CrossCut(_iBond) * Bonds().NormalStiffness(_iBond) * strainTotal);

	CVector3 dampingForceNorm = -mu * normalVelocity * Bonds().CrossCut(_iBond) * Bonds().NormalStiffness(_iBond) * fabs(strainTotal);
	if (dampingForceNorm.Length() > 0.5 * normalForce.Length())
		dampingForceNorm *= 0.5 * normalForce.Length() / dampingForceNorm.Length();

	_bonds.TangentialOverlap(_iBond) = M * Bonds().TangentialOverlap(_iBond) - tangentialVelocity * _timeStep;
	_bonds.TangentialForce(_iBond) = Bonds().TangentialOverlap(_iBond) * (Bonds().TangentialStiffness(_iBond) * Bonds().CrossCut(_iBond) / Bonds().InitialLength(_iBond));

	CVector3 dampingForceTang = -mu * tangentialVelocity * Bonds().TangentialOverlap(_iBond).Length() * (Bonds().TangentialStiffness(_iBond) * Bonds().CrossCut(_iBond) / Bonds().InitialLength(_iBond));
	if (dampingForceTang.Length() > 0.5 * Bonds().TangentialForce(_iBond).Length())
		dampingForceTang *= 0.5 * Bonds().TangentialForce(_iBond).Length() / dampingForceTang.Length();

	_bonds.NormalMoment(_iBond) = M * Bonds().NormalMoment(_iBond) - normalAngleVel * (_timeStep * 2 * Bonds().AxialMoment(_iBond) * Bonds().TangentialStiffness(_iBond) / Bonds().InitialLength(_iBond));
	_bonds.TangentialMoment(_iBond) = M * Bonds().TangentialMoment(_iBond) - tangAngleVel * (_timeStep * Bonds().NormalStiffness(_iBond) * Bonds().AxialMoment(_iBond) / Bonds().InitialLength(_iBond));
	_bonds.TotalForce(_iBond) = normalForce + Bonds().TangentialForce(_iBond) + dampingForceNorm + dampingForceTang;

	_bonds.UnsymMoment(_iBond) = currentBond * 0.5 * Bonds().TangentialForce(_iBond);
	_bonds.PrevBond(_iBond) = currentBond;

	if (m_parameters[0].value == 0.0) return; // consider breakage

	// check the bond destruction
	double forceLength = normalForce.Length();
	if (strainTotal <= 0)	// compression
		forceLength *= -1;

	const double maxStress = forceLength / Bonds().CrossCut(_iBond) + Bonds().TangentialMoment(_iBond).Length() * Bonds().Diameter(_iBond) / (2 * Bonds().AxialMoment(_iBond));
	const double maxTorque = Bonds().TangentialForce(_iBond).Length() / Bonds().CrossCut(_iBond) + Bonds().NormalMoment(_iBond).Length() * Bonds().Diameter(_iBond) / (2 * 2 * Bonds().AxialMoment(_iBond));

	if (maxStress >= Bonds().NormalStrength(_iBond) || maxTorque >= Bonds().TangentialStrength(_iBond))
	{
		_bonds.Active(_iBond) = false;
		_bonds.EndActivity(_iBond) = _time;
		*_brokenBondsNum += 1;
	}
}

void CModelSBKelvin::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
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
