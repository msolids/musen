/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWJKR.h"

CModelPWJKR::CModelPWJKR()
{
	m_name         = "JKR";
	m_uniqueKey    = "470A5893541D4204A7F1F2C993C667FB";
	m_helpFileName = "/Contact Models/JKR.pdf";
	m_hasGPUSupport = true;
}

void CModelPWJKR::CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _collision) const
{
	const double   partRadius  = Particles().Radius(_iPart);
	const CVector3 partAnglVel = Particles().AnglVel(_iPart);
	const CVector3 normVector  = Walls().NormalVector(_iWall);

	const CVector3 rc          = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _collision->vContactVector;
	const double   rcLen       = rc.Length();
	const CVector3 rcNorm      = rc / rcLen;

	// normal overlap
	const double normOverlap =  partRadius - rcLen;
	if (normOverlap < 0) return;

	// normal and tangential relative velocity
	const CVector3 rotVel        = !Walls().RotVel(_iWall).IsZero() ? (_collision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };
	const CVector3 relVel        = Particles().Vel(_iPart) - Walls().Vel(_iWall) + rotVel + rcNorm * partAnglVel * partRadius;
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;

	// radius of the contact area
	const double contactAreaRadius = std::sqrt(partRadius * normOverlap);

	// normal force with damping
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius;
	const double normContactForceLen = 4. * std::pow(contactAreaRadius, 3.) * _interactProp.dEquivYoungModulus / (3. * partRadius) -
		std::sqrt(8. * PI * _interactProp.dEquivYoungModulus * _interactProp.dEquivSurfaceEnergy * std::pow(contactAreaRadius, 3.)) *
		std::abs(DotProduct(rcNorm, normVector));
	const double normDampingForceLen = _2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * std::sqrt(Kn * Particles().Mass(_iPart));
	const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

	// rotate old tangential overlap
	CVector3 tangOverlapRot = _collision->vTangOverlap - normVector * DotProduct(normVector, _collision->vTangOverlap);
	if (tangOverlapRot.IsSignificant())
		tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
	// calculate new tangential overlap
	CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

	// tangential force with damping
	const double Kt = 8 * _interactProp.dEquivShearModulus * contactAreaRadius;
	const CVector3 tangShearForce = -Kt * tangOverlap;
	const CVector3 tangDampingForce = tangRelVel * (_2_SQRT_5_6 * _interactProp.dAlpha * std::sqrt(Kt *  Particles().Mass(_iPart)));

	// check slipping condition and calculate total tangential force
	CVector3 tangForce;
	const double tangShearForceLen = tangShearForce.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * std::abs(normContactForceLen + normDampingForceLen);
	if (tangShearForceLen > frictionForceLen)
	{
		tangForce   = tangShearForce * frictionForceLen / tangShearForceLen;
		tangOverlap = tangForce / -Kt;
	}
	else
		tangForce   = tangShearForce + tangDampingForce;

	// rolling torque
	const CVector3 rollingTorque = partAnglVel.IsSignificant() ? partAnglVel * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * partRadius / partAnglVel.Length()) : CVector3{ 0 };

	// final forces and moments
	const CVector3 totalForce = normForce + tangForce;
	const CVector3 moment     = normVector * tangForce * -partRadius + rollingTorque;

	// store results in collision
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce;
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment;
}
