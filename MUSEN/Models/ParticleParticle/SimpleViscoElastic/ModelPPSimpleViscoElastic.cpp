/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSimpleViscoElastic.h"

CModelPPSimpleViscoElastic::CModelPPSimpleViscoElastic()
{
	m_name          = "Simple viscoelastic";
	m_uniqueKey     = "5B1DBC037BAE488086159B7731E1D68F";
	m_helpFileName  = "/Contact Models/SimpleViscoElastic.pdf";
	m_hasGPUSupport = true;

	/* 0*/ AddParameter("NORMAL_FORCE_COEFF"      , "Coefficient of normal force", 1);
	/* 1*/ AddParameter("NORMAL_DAMPING_PARAMETER", "Damping parameter"          , 0);
}

void CModelPPSimpleViscoElastic::CalculatePP(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double Kn = m_parameters[0].value;
	const double mu = m_parameters[1].value;

	const CVector3 normVector = _collision->vContactVector.Normalized();

	// relative velocity (normal)
	const CVector3 relVel      = Particles().Vel(_iDst) - Particles().Vel(_iSrc);
	const double normRelVelLen = DotProduct(normVector, relVel);

	// normal force with damping
	const double normContactForceLen = -_collision->dNormalOverlap * Kn;
	const double normDampingForceLen = mu * normRelVelLen;
	const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

	// store results in collision
	_collision->vTotalForce = normForce;
}

void CModelPPSimpleViscoElastic::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) += _collision->vTotalForce;
}

void CModelPPSimpleViscoElastic::ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) -= _collision->vTotalForce;
}
