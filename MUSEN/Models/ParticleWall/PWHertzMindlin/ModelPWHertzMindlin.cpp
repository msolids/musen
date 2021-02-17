/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHertzMindlin.h"

CModelPWHertzMindlin::CModelPWHertzMindlin()
{
	m_name = "Hertz-Mindlin";
	m_uniqueKey = "906949ACFFAE4B8C8B1B65509930EA6D";
	m_helpFileName = "/Contact Models/HertzMindlin.pdf";

	m_hasGPUSupport = true;
}

void CModelPWHertzMindlin::CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dPartRadius = Particles().Radius(_iPart);
	const CVector3 partAnglVel = Particles().AnglVel(_iPart);

	const CVector3 vRc          = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _pCollision->vContactVector;
	const double   dRc          = vRc.Length();
	const CVector3 nRc          = vRc / dRc; // = vRc.Normalized()
	const CVector3 vVelDueToRot = !Walls().RotVel(_iWall).IsZero() ? (_pCollision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };

	// normal overlap
	const double dNormalOverlap =  dPartRadius - dRc;
	if (dNormalOverlap < 0) return;

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       =  Particles().Vel(_iPart) - Walls().Vel(_iWall) + vVelDueToRot + nRc * partAnglVel * dPartRadius;
	const double   dRelVelNormal = DotProduct(Walls().NormalVector(_iWall), vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * Walls().NormalVector(_iWall);
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// normal force with damping
	const double Kn = 2 * _interactProp.dEquivYoungModulus * std::sqrt( dPartRadius * dNormalOverlap);
	const double dDampingForce = 1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn *  Particles().Mass(_iPart));
	const double dNormalForce  = 2. / 3.  * dNormalOverlap * Kn * std::abs(DotProduct(nRc, Walls().NormalVector(_iWall)));

	// increment of tangential force with damping
	const double Kt = 8. * _interactProp.dEquivShearModulus * std::sqrt(dPartRadius * dNormalOverlap);
	const CVector3 vDampingTangForce = vRelVelTang * (1.8257 * _interactProp.dAlpha * std::sqrt(Kt *  Particles().Mass(_iPart)));

	// rotate old tangential force
	CVector3 vTangOverlap = _pCollision->vTangOverlap - Walls().NormalVector(_iWall) * DotProduct(Walls().NormalVector(_iWall), _pCollision->vTangOverlap);
	if (vTangOverlap.IsSignificant())
		vTangOverlap = vTangOverlap * _pCollision->vTangOverlap.Length() / vTangOverlap.Length();

	_pCollision->vTangOverlap = vTangOverlap + vRelVelTang * _timeStep;
	CVector3 vTangForce = -Kt * _pCollision->vTangOverlap;
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
	const CVector3 vRollingTorque = partAnglVel.IsSignificant() ? // if it is not zero, but small enough, its Length() can turn into zero and division fails
		partAnglVel * (-_interactProp.dRollingFriction * std::abs(dNormalForce) * dPartRadius / partAnglVel.Length()) : CVector3{ 0 };

	// store results in collision
	_pCollision->vTangForce     = vTangForce;
	_pCollision->vTotalForce    = _pCollision->vTangForce + (dNormalForce + dDampingForce) * Walls().NormalVector(_iWall);
	_pCollision->vResultMoment1 = Walls().NormalVector(_iWall) * _pCollision->vTangForce * -dPartRadius + vRollingTorque;
}
