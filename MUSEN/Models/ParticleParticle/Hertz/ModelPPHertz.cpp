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
	const CVector3 srcAnglVel   = Particles().AnglVel(_iSrc);
	const CVector3 dstAnglVel   = Particles().AnglVel(_iDst);
	const double dPartSrcRadius = Particles().Radius(_iSrc);
	const double dPartDstRadius = Particles().Radius(_iDst);

	const CVector3 vRcSrc        = _pCollision->vContactVector * ( dPartSrcRadius / (dPartSrcRadius + dPartDstRadius));
	const CVector3 vRcDst        = _pCollision->vContactVector * (-dPartDstRadius / (dPartSrcRadius + dPartDstRadius));
	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       = Particles().Vel(_iDst) + dstAnglVel * vRcDst - (Particles().Vel(_iSrc) + srcAnglVel * vRcSrc);
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
	const CVector3 vRollingTorque1 = srcAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		srcAnglVel * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * dPartSrcRadius / srcAnglVel.Length()) : CVector3{ 0 };
	const CVector3 vRollingTorque2 = dstAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		dstAnglVel * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * dPartDstRadius / dstAnglVel.Length()) : CVector3{ 0 };

	// calculate moment of TangForce
	const CVector3 vecMoment1 = vNormalVector * newTangForce * dPartSrcRadius + vRollingTorque1;
	const CVector3 vecMoment2 = vNormalVector * newTangForce * dPartDstRadius + vRollingTorque2;

	_pCollision->vTotalForce    = vNormalVector * dNormalForce + newTangForce;
	_pCollision->vResultMoment1 = vecMoment1;
	_pCollision->vResultMoment2 = vecMoment2;
}