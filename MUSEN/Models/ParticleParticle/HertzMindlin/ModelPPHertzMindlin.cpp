/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPHertzMindlin.h"

CModelPPHertzMindlin::CModelPPHertzMindlin()
{
	m_name          = "Hertz-Mindlin";
	m_uniqueKey     = "B18A46C2786D4D44B925A8A04D0D1008";
	m_helpFileName  = "/Contact Models/HertzMindlin.pdf";
	m_hasGPUSupport = true;
}

void CModelPPHertzMindlin::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	const CVector3 anglVel1 = Particles().AnglVel(_iSrc);
	const CVector3 anglVel2 = Particles().AnglVel(_iDst);
	const double   radius1  = Particles().Radius(_iSrc);
	const double   radius2  = Particles().Radius(_iDst);

	const CVector3 rc1        = _collision->vContactVector * ( radius1 / (radius1 + radius2));
	const CVector3 rc2        = _collision->vContactVector * (-radius2 / (radius1 + radius2));
	const CVector3 normVector = _collision->vContactVector.Normalized();

	// normal and tangential relative velocity
	const CVector3 relVel        = (Particles().Vel(_iDst) + anglVel2 * rc2) - (Particles().Vel(_iSrc) + anglVel1 * rc1);
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;

	// radius of the contact area
	const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * _collision->dNormalOverlap);

	// normal force with damping
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius;
	const double normContactForceLen = -_collision->dNormalOverlap * Kn * 2. / 3.;
	const double normDampingForceLen = -_2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * std::sqrt(Kn * _collision->dEquivMass);
	const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

	// rotate old tangential overlap
	CVector3 tangOverlapRot = _collision->vTangOverlap - normVector * DotProduct(normVector, _collision->vTangOverlap);
	if (tangOverlapRot.IsSignificant())
		tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
	// calculate new tangential overlap
	CVector3 tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

	// tangential force with damping
	const double Kt = 8 * _interactProp.dEquivShearModulus * contactAreaRadius;
	const CVector3 tangShearForce = tangOverlap * Kt;
	const CVector3 tangDampingForce = tangRelVel * (-_2_SQRT_5_6 * _interactProp.dAlpha * std::sqrt(Kt * _collision->dEquivMass));

	// check slipping condition and calculate total tangential force
	CVector3 tangForce;
	const double tangShearForceLen = tangShearForce.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * std::abs(normContactForceLen + normDampingForceLen);
	if (tangShearForceLen > frictionForceLen)
	{
		tangForce   = tangShearForce * frictionForceLen / tangShearForceLen;
		tangOverlap = tangForce / Kt;
	}
	else
		tangForce   = tangShearForce + tangDampingForce;

	// rolling torque
	const CVector3 rollingTorque1 = anglVel1.IsSignificant() ? anglVel1 * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * radius1 / anglVel1.Length()) : CVector3{ 0 };
	const CVector3 rollingTorque2 = anglVel2.IsSignificant() ? anglVel2 * (-_interactProp.dRollingFriction * std::abs(normContactForceLen) * radius2 / anglVel2.Length()) : CVector3{ 0 };

	// final forces and moments
	const CVector3 totalForce = normForce + tangForce;
	const CVector3 moment1    = normVector * tangForce * radius1 + rollingTorque1;
	const CVector3 moment2    = normVector * tangForce * radius2 + rollingTorque2;

	// store results in collision
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce;
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment1;
	_collision->vResultMoment2 = moment2;
}

void CModelPPHertzMindlin::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPPHertzMindlin::ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) -= _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment2;
}
