/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPChealNess.h"

/* This model is a combination between Hertz-Mindling model and model proposed by Cheal and Ness
https://doi.org/10.1122/1.5004007
https://doi.org/10.1016/j.powtec.2021.01.071  */


CModelPPChealNess::CModelPPChealNess()
{
	m_name          = "Cheal-Ness";
	m_uniqueKey     = "B18A4SSSS6D4D44B925A8A04D0D1008";
	m_hasGPUSupport = true;

	/* 0*/ AddParameter("MIN_THICKNESS"  , "Minimal liquid thickness [m]", 1e-5);
	/* 1*/ AddParameter("MAX_THICKNESS"  , "Maximal liquid thickness [m]", 5e-3);
	/* 2*/ AddParameter("FLUID_VISCOSITY", "Fluid viscosity [Pa*s]"      , 0.1 );
}

void CModelPPChealNess::CalculatePP(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double minThickness  = m_parameters[0].value;
	const double maxThickness  = m_parameters[1].value;
	const double fluidVisosity = m_parameters[2].value;

	const CVector3 anglVel1 = Particles().AnglVel(_iSrc);
	const CVector3 anglVel2 = Particles().AnglVel(_iDst);
	const double   radius1  = Particles().Radius(_iSrc);
	const double   radius2  = Particles().Radius(_iDst);

	const CVector3 rc1        = _collision->vContactVector * ( radius1 / (radius1 + radius2));
	const CVector3 rc2        = _collision->vContactVector * (-radius2 / (radius1 + radius2));
	const CVector3 normVector = _collision->vContactVector.Normalized();

	// adjusted normal overlap
	const double surfaceDistance = (Particles().Coord(_iDst) - Particles().Coord(_iSrc)).Length() - radius1 - radius2;
	const double normOverlap = surfaceDistance < 0 ? fabs(surfaceDistance) : 0.0;

	// normal and tangential relative velocity
	const CVector3 relVel        = (Particles().Vel(_iDst) + anglVel2 * rc2) - (Particles().Vel(_iSrc) + anglVel1 * rc1);
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normLenVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normLenVel;

	// radius of the contact area
	const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * normOverlap);

	// normal force with damping
	const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius;
	const double normContactForceLen = -normOverlap * Kn * 2. / 3.;
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

	// calculate lubrication force
	CVector3 normLubrForce{ 0 };
	CVector3 tangLubrForce{ 0 };
	const double dh = _collision->vContactVector.Length() - radius1 - radius2;
	if (minThickness <= dh && dh <= maxThickness)
	{
		double beta = radius2 / radius1;
		double Xi   = 2 * dh / (radius2 + radius1);
		double X11A = 6 * PI * radius1 * (2 * beta * beta / ((1 + pow(beta, 3)) * Xi) + beta * (1 + 7 * beta + beta * beta) / (5 * pow(1 + beta, 3)) * log(1 / Xi));
		double Y11A = 6 * PI * radius1 * (4 * beta * (1 + 7 * beta + beta * beta) / (15 * pow(1 + beta, 3)) * log(1 / Xi));
		double Y11B = -4 * PI * pow(radius1, 2) * (beta * (4 + beta) / (5 * pow(1 + beta, 2)) * log(1 / Xi));
		double Y11C = 8 * PI * pow(radius1, 3) * (2 * beta / (5 * (1 + beta)) * log(1 / Xi));
		double Y12C = Y11C / 4;
		double Y21B = -4 * PI * pow(radius2, 2) * ((4 + 1 / beta) / (5 * pow(1 + 1 / beta, 2)) * log(1 / Xi));
		CVector3 Nij       = -1 * normVector;
		CMatrix3 outerProd = OuterProduct(Nij, Nij);
		CVector3 velij     = Particles().Vel(_iSrc) - Particles().Vel(_iDst);
		normLubrForce = -1 * fluidVisosity * ((X11A * outerProd + Y11A * (CMatrix3::Identity() - outerProd)) * velij + Y11B * (anglVel1 * Nij) + Y21B * (anglVel2 * Nij));
		tangLubrForce = -1 * fluidVisosity * (Y11B * velij * Nij - (CMatrix3::Identity() - outerProd) * (Y11C * anglVel1 + Y12C * anglVel2));
	}

	// final forces and moments
	const CVector3 totalForce = normForce + tangForce + normLubrForce + tangLubrForce;
	const CVector3 moment1    = normVector * (tangForce + tangLubrForce) * radius1 + rollingTorque1;
	const CVector3 moment2    = normVector * (tangForce + tangLubrForce) * radius2 + rollingTorque2;

	// store results in collision
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce;
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment1;
	_collision->vResultMoment2 = moment2;
}

void CModelPPChealNess::ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPPChealNess::ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) -= _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment2;
}
