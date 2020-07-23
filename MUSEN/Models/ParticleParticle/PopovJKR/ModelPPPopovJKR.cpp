/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPPopovJKR.h"

CModelPPPopovJKR::CModelPPPopovJKR()
{
	m_name = "Popov-JKR";
	m_uniqueKey = "7A4C0DAF1F6D44AE925F4DD84563E36F";
	m_helpFileName = "/Contact Models/PopovJKR.pdf";
}

void CModelPPPopovJKR::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const CVector3 vRcSrc        = _pCollision->vContactVector * ( Particles().Radius(_iSrc) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vRcDst        = _pCollision->vContactVector * (-Particles().Radius(_iDst) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       = Particles().Vel(_iDst) - Particles().Vel(_iSrc) + vRcSrc * Particles().AnglVel(_iSrc) - vRcDst * Particles().AnglVel(_iDst);
	const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// normal force with damping
	const double dAdhForce = 3. / 2. * _interactProp.dEquivSurfaceTension * PI * _pCollision->dEquivRadius;
	const double dSadh = std::pow(0.4626 * std::pow(_interactProp.dEquivSurfaceTension / _interactProp.dEquivYoungModulus, 2) * _pCollision->dEquivRadius, 1. /3.);
	const double dNormalForce = -dAdhForce * (-1 + 0.12 * std::pow(_pCollision->dNormalOverlap / dSadh + 1, 5. / 3.));
	const double Kn = 2 * _interactProp.dEquivYoungModulus * std::sqrt(_pCollision->dEquivRadius * _pCollision->dNormalOverlap);
	const double dDampingForce = -1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn * _pCollision->dEquivMass);

	// increment of tangential force with damping
	const double Kt = 8 * _interactProp.dEquivShearModulus * std::sqrt(_pCollision->dEquivRadius * _pCollision->dNormalOverlap);
	const CVector3 vDampingTangForce = vRelVelTang * (-1.8257 * _interactProp.dAlpha * std::sqrt(Kt * _pCollision->dEquivMass));

	// rotate old tangential force
	CVector3 vTangOverlap = _pCollision->vTangOverlap - vNormalVector * DotProduct(vNormalVector, _pCollision->vTangOverlap);
	if (vTangOverlap.IsSignificant())
		vTangOverlap = vTangOverlap * _pCollision->vTangOverlap.Length() / vTangOverlap.Length();
	_pCollision->vTangOverlap = vTangOverlap + vRelVelTang * _timeStep;

	CVector3 vTangForce = _pCollision->vTangOverlap * Kt;

	// check slipping condition
	const double dNewTangForce = vTangForce.Length();
	if (dNewTangForce > _interactProp.dSlidingFriction * std::abs(dNormalForce))
	{
		vTangForce = vTangForce * _interactProp.dSlidingFriction * std::abs(dNormalForce) / dNewTangForce;
		_pCollision->vTangOverlap = vTangForce / Kt;
	}
	// save tangential force
	_pCollision->vTangForce = vTangForce + vDampingTangForce;

	// calculate rolling friction
	const CVector3 vRollingTorque1 = Particles().AnglVel(_iSrc).IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		Particles().AnglVel(_iSrc) * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * Particles().Radius(_iSrc) / Particles().AnglVel(_iSrc).Length()) : CVector3{ 0 };
	const CVector3 vRollingTorque2 = Particles().AnglVel(_iDst).IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		Particles().AnglVel(_iDst) * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * Particles().Radius(_iDst) / Particles().AnglVel(_iDst).Length()) : CVector3{ 0 };

	// calculate moment of TangForce
	const CVector3 vMoment1 = vNormalVector * _pCollision->vTangForce*Particles().Radius(_iSrc) + vRollingTorque1;
	const CVector3 vMoment2 = vNormalVector * _pCollision->vTangForce*Particles().Radius(_iDst) + vRollingTorque2;

	_pCollision->vTotalForce    = vNormalVector * (dNormalForce + dDampingForce) + _pCollision->vTangForce;
	_pCollision->vResultMoment1 = vMoment1;
	_pCollision->vResultMoment2 = vMoment2;
}