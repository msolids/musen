/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWJKR.h"

CModelPWJKR::CModelPWJKR()
{
	m_name = "JKR";
	m_uniqueKey = "470A5893541D4204A7F1F2C993C667FB";
	m_helpFileName = "/Contact Models/JKR.pdf";
}

void CModelPWJKR::CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dPartRadius   = Particles().Radius(_iPart);
	const CVector3 partAnglVel = Particles().AnglVel(_iPart);

	const CVector3 vRc          = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _pCollision->vContactVector;
	const double   dRc          = vRc.Length();
	const CVector3 nRc          = vRc / dRc; // = vRc.Normalized()
	const CVector3 vVelDueToRot = !Walls().RotVel(_iWall).IsZero() ? (_pCollision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       =  Particles().Vel(_iPart) - Walls().Vel(_iWall) + vVelDueToRot + nRc *  partAnglVel *  dPartRadius;
	const double   dRelVelNormal = DotProduct(Walls().NormalVector(_iWall), vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * Walls().NormalVector(_iWall);
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// normal and tangential overlaps
	const double dNormalOverlap =  dPartRadius - dRc;
	if (dNormalOverlap < 0) return;
	const CVector3 vDeltaTangOverlap = vRelVelTang * _timeStep;

	const double Kn = 2 * _interactProp.dEquivYoungModulus * std::sqrt( dPartRadius * dNormalOverlap);
	const double dDampingForce = 1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn *  Particles().Mass(_iPart));

	const double dContactRadius = std::sqrt(dNormalOverlap * Particles().ContactRadius(_iPart));
	const double dNormalForce = 4. * std::pow(dContactRadius, 3.) * _interactProp.dEquivYoungModulus / (3. * Particles().ContactRadius(_iPart)) -
		std::sqrt(8. * PI * _interactProp.dEquivYoungModulus * _interactProp.dEquivSurfaceEnergy * std::pow(dContactRadius, 3.)) *
		std::abs(DotProduct(nRc, Walls().NormalVector(_iWall)));

	// increment of tangential force with damping
	const double Kt = 8 * _interactProp.dEquivShearModulus * std::sqrt(dPartRadius * dNormalOverlap);
	const CVector3 vDampingTangForce = vRelVelTang * (1.8257 * _interactProp.dAlpha * std::sqrt(Kt *  Particles().Mass(_iPart)));

	// rotate old tangential force
	CVector3 vTangOverlap = _pCollision->vTangOverlap - Walls().NormalVector(_iWall) * DotProduct(Walls().NormalVector(_iWall), _pCollision->vTangOverlap);
	if (vTangOverlap.IsSignificant())
		vTangOverlap = vTangOverlap * _pCollision->vTangOverlap.Length() / vTangOverlap.Length();

	_pCollision->vTangOverlap = vTangOverlap + vDeltaTangOverlap;
	CVector3 vTangForce = _pCollision->vTangOverlap * -Kt;
	// check slipping condition
	const double dNewTangForce = vTangForce.Length();
	if (dNewTangForce > _interactProp.dSlidingFriction * std::abs(dNormalForce))
	{
		vTangForce *= _interactProp.dSlidingFriction * std::abs(dNormalForce) / dNewTangForce;
		_pCollision->vTangOverlap = vTangForce / -Kt;
	}
	else
		vTangForce += vDampingTangForce;

	// calculate rolling friction
	const CVector3 vRollingTorque =  partAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		partAnglVel * (-_interactProp.dRollingFriction * std::abs(dNormalForce) *  dPartRadius /  partAnglVel.Length()) : CVector3{ 0 };

	// calculate moment
	const CVector3 vMoment = Walls().NormalVector(_iWall) * _pCollision->vTangForce * - dPartRadius + vRollingTorque;

	// store results in collision
	_pCollision->vTangForce     = vTangForce;
	_pCollision->vTotalForce    = _pCollision->vTangForce + (dNormalForce + dDampingForce) * Walls().NormalVector(_iWall);
	_pCollision->vResultMoment1 = vMoment;
}
