/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSimpleViscoElastic.h"

CModelPPSimpleViscoElastic::CModelPPSimpleViscoElastic()
{
	m_name = "Simple viscoelastic";
	m_uniqueKey = "5B1DBC037BAE488086159B7731E1D68F";
	m_helpFileName = "/Contact Models/SimpleViscoElastic.pdf";

	AddParameter("NORMAL_FORCE_COEFF", "Coefficient of normal force", 1);
	AddParameter("NORMAL_DAMPING_PARAMETER", "Damping parameter", 0);

	m_hasGPUSupport = true;
}

void CModelPPSimpleViscoElastic::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dKn = m_parameters[0].value;
	const double dMu = m_parameters[1].value;

	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();
	_pCollision->vTotalForce = vNormalVector * (-_pCollision->dNormalOverlap * dKn);

	if (dMu != 0)
	{
		// relative velocity (normal)
		const CVector3 vRelVelocity =  Particles().Vel(_iDst) -  Particles().Vel(_iSrc);
		const double dRelVelNormal  = DotProduct(vNormalVector, vRelVelocity);

		// normal force with damping
		const double dDampingForce = -dMu * dRelVelNormal;

		_pCollision->vTotalForce += vNormalVector * dDampingForce;
	}
}
