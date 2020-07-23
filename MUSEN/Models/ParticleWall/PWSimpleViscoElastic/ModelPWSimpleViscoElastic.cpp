/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWSimpleViscoElastic.h"

CModelPWSimpleViscoElastic::CModelPWSimpleViscoElastic()
{
	m_name = "Simple viscoelastic";
	m_uniqueKey = "522836B77298445A8FFD02CA55FFF717";
	m_helpFileName = "/Contact Models/SimpleViscoElastic.pdf";

	AddParameter("NORMAL_FORCE_COEFF", "Coefficient of normal force", 1);
	AddParameter("NORMAL_DAMPING_PARAMETER", "Damping parameter", 0);

	m_hasGPUSupport = true;
}

void CModelPWSimpleViscoElastic::CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dKn = m_parameters[0].value;
	const double dMu = m_parameters[1].value;

	const CVector3 vRc = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _pCollision->vContactVector;
	const double   dRc = vRc.Length();
	const CVector3 nRc = vRc / dRc; // = vRc.Normalized()

	// normal and Tangential overlaps
	const double dNormalOverlap =  Particles().Radius(_iPart) - dRc;
	if (dNormalOverlap < 0) return;

	_pCollision->vTotalForce = Walls().NormalVector(_iWall) * (dNormalOverlap * dKn * std::abs(DotProduct(nRc, Walls().NormalVector(_iWall))));

	if (dMu != 0)
	{
		const CVector3 vVelDueToRot = !Walls().RotVel(_iWall).IsZero() ? (_pCollision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };

		// relative velocity (normal and tangential)
		const CVector3 vRelVel     =  Particles().Vel(_iPart) - Walls().Vel(_iWall) + vVelDueToRot;
		const double dRelVelNormal = DotProduct(Walls().NormalVector(_iWall), vRelVel);
		// normal force with damping
		const double dDampingForce = dMu * dRelVelNormal;

		_pCollision->vTotalForce += dDampingForce * Walls().NormalVector(_iWall);
	}
}
