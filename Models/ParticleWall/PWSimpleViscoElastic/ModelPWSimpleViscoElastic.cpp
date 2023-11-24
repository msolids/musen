/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWSimpleViscoElastic.h"

CModelPWSimpleViscoElastic::CModelPWSimpleViscoElastic()
{
	m_name          = "Simple viscoelastic";
	m_uniqueKey     = "522836B77298445A8FFD02CA55FFF717";
	m_helpFileName  = "/Contact Models/SimpleViscoElastic.pdf";
	m_hasGPUSupport = true;

	/* 0*/ AddParameter("NORMAL_FORCE_COEFF"      , "Coefficient of normal force", 1);
	/* 1*/ AddParameter("NORMAL_DAMPING_PARAMETER", "Damping parameter"          , 0);
}

void CModelPWSimpleViscoElastic::CalculatePW(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double Kn = m_parameters[0].value;
	const double mu = m_parameters[1].value;

	const CVector3 normVector = Walls().NormalVector(_iWall);

	const CVector3 rc     = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _collision->vContactVector;
	const double   rcLen  = rc.Length();
	const CVector3 rcNorm = rc / rcLen;

	// normal overlap
	const double normOverlap =  Particles().Radius(_iPart) - rcLen;
	if (normOverlap < 0) return;

	// normal and tangential relative velocity
	const CVector3 rotVel      = !Walls().RotVel(_iWall).IsZero() ? (_collision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };
	const CVector3 relVel      = Particles().Vel(_iPart) - Walls().Vel(_iWall) + rotVel;
	const double normRelVelLen = DotProduct(normVector, relVel);

	// normal force with damping
	const double normContactForceLen = normOverlap * Kn * std::abs(DotProduct(rcNorm, normVector));
	const double normDampingForceLen = -mu * normRelVelLen;
	const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

	// store results in collision
	_collision->vTotalForce = normForce;
}

void CModelPWSimpleViscoElastic::ConsolidatePart(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) += _collision->vTotalForce;
}

void CModelPWSimpleViscoElastic::ConsolidateWall(double _time, double _timeStep, size_t _iWall, SWallStruct& _walls, const SCollision* _collision) const
{
	_walls.Force(_iWall) -= _collision->vTotalForce;
}