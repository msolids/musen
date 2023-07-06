/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPHertz.h"

CModelPPHertz::CModelPPHertz()
{
	m_name          = "Hertz";
	m_uniqueKey     = "B7CBEB0657884100930E6C68E2C438EB";
	m_helpFileName  = "/Contact Models/Hertz.pdf";
	m_hasGPUSupport = false;
}

void CModelPPHertz::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	const CVector3 anglVel1 = Particles().AnglVel(_iSrc);
	const CVector3 anglVel2 = Particles().AnglVel(_iDst);
	const double   radius1  = Particles().Radius(_iSrc);
	const double   radius2  = Particles().Radius(_iDst);

	const CVector3 rc1        = _collision->vContactVector * ( radius1 / (radius1 + radius2));
	const CVector3 rc2        = _collision->vContactVector * (-radius2 / (radius1 + radius2));
	const CVector3 normVector = _collision->vContactVector.Normalized();

	// normal and tangential relative velocity
	const CVector3 relVel        = Particles().Vel(_iDst) + anglVel2 * rc2 - (Particles().Vel(_iSrc) + anglVel1 * rc1);
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;

	// normal and tangential overlaps
	const double normOverlap = _collision->dNormalOverlap;
	const CVector3 tangOverlap = tangRelVel * _timeStep;

	// radius of the contact area
	const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * normOverlap);

	// normal force
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius;
	const double normForceLen = -2. / 3. * normOverlap * Kn;
	const CVector3 normForce = normVector * normForceLen;

	// rotate old tangential overlap
	CVector3 tangOverlapRot = _collision->vTangForce - normVector * DotProduct(normVector, _collision->vTangForce);
	if (tangOverlapRot.IsSignificant())
		tangOverlapRot *= _collision->vTangForce.Length() / tangOverlapRot.Length();

	// tangential force
	const double Kt = 8 * _interactProp.dEquivShearModulus * contactAreaRadius;
	CVector3 tangForce = tangOverlapRot + tangOverlap * Kt;

	// check slipping condition and calculate total tangential force
	const double tangForceLen = tangForce.Length();
	const double frictionForceLen = _interactProp.dSlidingFriction * std::abs(normForceLen);
	if (tangForceLen > frictionForceLen)
		tangForce *= frictionForceLen / tangForceLen;

	// rolling torque
	const CVector3 rollingTorque1 = anglVel1.IsSignificant() ? anglVel1 * (-_interactProp.dRollingFriction * std::abs(normForceLen) * radius1 / anglVel1.Length()) : CVector3{ 0 };
	const CVector3 rollingTorque2 = anglVel2.IsSignificant() ? anglVel2 * (-_interactProp.dRollingFriction * std::abs(normForceLen) * radius2 / anglVel2.Length()) : CVector3{ 0 };

	// final forces and moments
	const CVector3 totalForce = normForce + tangForce;
	const CVector3 moment1    = normVector * tangForce * radius1 + rollingTorque1;
	const CVector3 moment2    = normVector * tangForce * radius2 + rollingTorque2;

	// store results in collision
	_collision->vTotalForce    = totalForce;
	_collision->vTangForce     = tangForce;
	_collision->vResultMoment1 = moment1;
	_collision->vResultMoment2 = moment2;
}

void CModelPPHertz::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPPHertz::ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) -= _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment2;
}
