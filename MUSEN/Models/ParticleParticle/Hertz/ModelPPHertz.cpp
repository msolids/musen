/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPHertz.h"

CModelPPHertz::CModelPPHertz()
{
	m_name = "Hertz";
	m_uniqueKey = "B7CBEB0657884100930E6C68E2C438EB";
	m_helpFileName = "/Contact Models/Hertz.pdf";
}

void CModelPPHertz::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const CVector3 vRcSrc        = _pCollision->vContactVector * ( Particles().Radius(_iSrc) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vRcDst        = _pCollision->vContactVector * (-Particles().Radius(_iDst) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       = Particles().Vel(_iDst) + Particles().AnglVel(_iDst) * vRcDst - (Particles().Vel(_iSrc) + Particles().AnglVel(_iSrc) * vRcSrc);
	const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// normal and tangential overlaps
	const double dNormalOverlap = _pCollision->dNormalOverlap;
	const CVector3 vTangOverlap = vRelVelTang * _timeStep;

	// normal force with damping
	const double Kn = 2 * _interactProp.dEquivYoungModulus * std::sqrt(_pCollision->dEquivRadius * dNormalOverlap);
	const double dNormalForce = -1. * 2. / 3. * dNormalOverlap * Kn;

	// increment of tangential force with damping
	const double Kt = 8 * _interactProp.dEquivShearModulus * std::sqrt(_pCollision->dEquivRadius * dNormalOverlap);
	const CVector3 vDeltaTangForce = vTangOverlap * Kt;

	// rotate old tangential force
	CVector3 vTangForceCor = _pCollision->vTangForce - vNormalVector * DotProduct(vNormalVector, _pCollision->vTangForce);
	if (vTangForceCor.IsSignificant())
		vTangForceCor *= _pCollision->vTangForce.Length() / vTangForceCor.Length();
	CVector3 newTangForce = vTangForceCor + vDeltaTangForce;

	// check slipping condition
	if (newTangForce.Length() > _interactProp.dSlidingFriction * std::abs(dNormalForce))
		newTangForce = newTangForce * _interactProp.dSlidingFriction * std::abs(dNormalForce) / newTangForce.Length();

	// save old tangential force
	_pCollision->vTangForce = newTangForce;

	// calculate rolling friction
	const CVector3 vRollingTorque1 = Particles().AnglVel(_iSrc).IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		Particles().AnglVel(_iSrc) * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * Particles().Radius(_iSrc) / Particles().AnglVel(_iSrc).Length()) : CVector3{ 0 };
	const CVector3 vRollingTorque2 = Particles().AnglVel(_iDst).IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		Particles().AnglVel(_iDst) * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * Particles().Radius(_iDst) / Particles().AnglVel(_iDst).Length()) : CVector3{ 0 };

	// calculate moment of TangForce
	const CVector3 vecMoment1 = vNormalVector * newTangForce * Particles().Radius(_iSrc) + vRollingTorque1;
	const CVector3 vecMoment2 = vNormalVector * newTangForce * Particles().Radius(_iDst) + vRollingTorque2;

	_pCollision->vTotalForce    = vNormalVector * dNormalForce + newTangForce;
	_pCollision->vResultMoment1 = vecMoment1;
	_pCollision->vResultMoment2 = vecMoment2;
}