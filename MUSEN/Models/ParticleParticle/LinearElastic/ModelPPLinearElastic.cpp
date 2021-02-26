/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPLinearElastic.h"

CModelPPLinearElastic::CModelPPLinearElastic()
{
	m_name = "Linear Elastic";
	m_uniqueKey = "B18A46C2786D4D44B925A8A04DAD1008";
	m_helpFileName = "/Contact Models/LinearElastic.pdf";

	AddParameter("Kn", "Normal stiffness [N/m]", 1e+3);
	AddParameter("Kt", "Tangential stiffness [N/m]", 1e+3);

	m_hasGPUSupport = true;
}

void CModelPPLinearElastic::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double Kn = m_parameters[0].value;
	const double Kt = m_parameters[1].value;

	const CVector3 vRcSrc        = _pCollision->vContactVector * ( Particles().Radius(_iSrc) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vRcDst        = _pCollision->vContactVector * (-Particles().Radius(_iDst) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       = Particles().Vel(_iDst) + Particles().AnglVel(_iDst) * vRcDst - (Particles().Vel(_iSrc) + Particles().AnglVel(_iSrc) * vRcSrc);
	const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// normal force with damping
	const double dDampingForce = -1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn * _pCollision->dEquivMass);
	const double dNormalForce  = -_pCollision->dNormalOverlap * Kn;

	// increment of tangential force with damping
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
	else
		_pCollision->vTangForce = vTangForce + vDampingTangForce;

	// calculate rolling friction
	const CVector3 vRollingTorque1 = Particles().AnglVel(_iSrc).IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		Particles().AnglVel(_iSrc) * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * Particles().Radius(_iSrc) / Particles().AnglVel(_iSrc).Length()) : CVector3{ 0 };
	const CVector3 vRollingTorque2 = Particles().AnglVel(_iDst).IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		Particles().AnglVel(_iDst) * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * Particles().Radius(_iDst) / Particles().AnglVel(_iDst).Length()) : CVector3{ 0 };

	// calculate moment of TangForce
	const CVector3 vMoment1 = vNormalVector * _pCollision->vTangForce * Particles().Radius(_iSrc) + vRollingTorque1;
	const CVector3 vMoment2 = vNormalVector * _pCollision->vTangForce * Particles().Radius(_iDst) + vRollingTorque2;

	_pCollision->vTotalForce    = vNormalVector * (dNormalForce + dDampingForce) + _pCollision->vTangForce;
	_pCollision->vResultMoment1 = vMoment1;
	_pCollision->vResultMoment2 = vMoment2;
}
