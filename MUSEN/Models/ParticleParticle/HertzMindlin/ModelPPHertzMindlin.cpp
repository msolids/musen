/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPHertzMindlin.h"

CModelPPHertzMindlin::CModelPPHertzMindlin()
{
	m_name = "Hertz-Mindlin";
	m_uniqueKey = "B18A46C2786D4D44B925A8A04D0D1008";
	m_helpFileName = "/Contact Models/HertzMindlin.pdf";

	m_hasGPUSupport = true;
}

void CModelPPHertzMindlin::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const CVector3 srcAnglVel   = Particles().AnglVel(_iSrc);
	const CVector3 dstAnglVel   = Particles().AnglVel(_iDst);
	const double dPartSrcRadius = Particles().Radius(_iSrc);
	const double dPartDstRadius = Particles().Radius(_iDst);

	const CVector3 vRcSrc        = _pCollision->vContactVector * (dPartSrcRadius / (dPartSrcRadius + dPartDstRadius));
	const CVector3 vRcDst        = _pCollision->vContactVector * (-dPartDstRadius / (dPartSrcRadius + dPartDstRadius));
	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       = Particles().Vel(_iDst) + dstAnglVel * vRcDst - (Particles().Vel(_iSrc) + srcAnglVel * vRcSrc);
	const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// parameter for fast calculation
	const double dTemp2 = std::sqrt(_pCollision->dEquivRadius * _pCollision->dNormalOverlap);

	// normal force with damping
	const double Kn = 2 * _interactProp.dEquivYoungModulus * dTemp2;
	const double dDampingForce = -1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn * _pCollision->dEquivMass);
	const double dNormalForce  = -_pCollision->dNormalOverlap * Kn * 2. / 3.;

	// increment of tangential force with damping
	const double Kt = 8 * _interactProp.dEquivShearModulus * dTemp2;
	const CVector3 vDampingTangForce = vRelVelTang * (-1.8257 * _interactProp.dAlpha * std::sqrt(Kt * _pCollision->dEquivMass));

	// rotate old tangential force
	CVector3 vTangOverlap = _pCollision->vTangOverlap - vNormalVector * DotProduct(vNormalVector, _pCollision->vTangOverlap);
	const double dTangOverlapSqrLen = vTangOverlap.SquaredLength();
	if (dTangOverlapSqrLen > 0)
		vTangOverlap = vTangOverlap * _pCollision->vTangOverlap.Length() / std::sqrt(dTangOverlapSqrLen);
	_pCollision->vTangOverlap = vTangOverlap + vRelVelTang * _timeStep;

	CVector3 vTangForce = _pCollision->vTangOverlap * Kt;

	// check slipping condition
	const double dNewTangForce = vTangForce.Length();
	if (dNewTangForce > _interactProp.dSlidingFriction * std::abs(dNormalForce))
	{
		vTangForce *= _interactProp.dSlidingFriction * std::abs(dNormalForce) / dNewTangForce;
		_pCollision->vTangOverlap = vTangForce / Kt;
	}
	else
		vTangForce += vDampingTangForce;

	// calculate rolling friction
	const CVector3 vRollingTorque1 = srcAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		srcAnglVel * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * dPartSrcRadius / srcAnglVel.Length()) : CVector3{ 0 };
	const CVector3 vRollingTorque2 = dstAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		dstAnglVel * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * dPartDstRadius / dstAnglVel.Length()) : CVector3{ 0 };

	// calculate moments
	const CVector3 vMoment1 = vNormalVector * vTangForce * dPartSrcRadius + vRollingTorque1;
	const CVector3 vMoment2 = vNormalVector * vTangForce * dPartDstRadius + vRollingTorque2;

	_pCollision->vTangForce     = vTangForce;
	_pCollision->vTotalForce    = vNormalVector * (dNormalForce + dDampingForce) + vTangForce;
	_pCollision->vResultMoment1 = vMoment1;
	_pCollision->vResultMoment2 = vMoment2;
}
