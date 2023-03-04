/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPHertzMindlinLiquid.h"

CModelPPHertzMindlinLiquid::CModelPPHertzMindlinLiquid()
{
	m_name          = "Hertz-MindlinLiquid";
	m_uniqueKey     = "B18A46C2786D4D44B925AASS4D0D1008";
	m_hasGPUSupport = false;

	/* 0*/ AddParameter("MIN_THICKNESS"    , "Min. thickness of liquid [m]", 1e-5);
	/* 1*/ AddParameter("CONTACT_ANGLE"    , "Contact angle [grad]"        , 70  );
	/* 2*/ AddParameter("SURFACE_TENSION"  , "Surface tension [N/m]"       , 0.7 );
	/* 3*/ AddParameter("DYNAMIC_VISCOSITY", "Dynamic viscosity [Pa*s]"    , 0.1 );
}

void CModelPPHertzMindlinLiquid::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double minThickness   = m_parameters[0].value;
	const double contactAngle   = m_parameters[1].value * PI / 180;
	const double surfaceTension = m_parameters[2].value;
	const double viscosity      = m_parameters[3].value;

	const CVector3 anglVel1 = Particles().AnglVel(_iSrc);
	const CVector3 anglVel2 = Particles().AnglVel(_iDst);
	const double   radius1  = Particles().Radius(_iSrc);
	const double   radius2  = Particles().Radius(_iDst);

	const double bondLength = std::max(_collision->vContactVector.Length() - radius1 - radius2, minThickness);
	const CVector3 rAC = -0.5 * _collision->vContactVector;

	const double realOverlap = _collision->dNormalOverlap - (Particles().ContactRadius(_iSrc) - radius1) - (Particles().ContactRadius(_iDst) - radius2);

	const CVector3 rc1        = _collision->vContactVector * ( radius1 / (radius1 + radius2));
	const CVector3 rc2        = _collision->vContactVector * (-radius2 / (radius1 + radius2));
	const CVector3 normVector = _collision->vContactVector.Normalized();

	// normal and tangential relative velocity
	const CVector3 relVel        = Particles().Vel(_iDst) - Particles().Vel(_iSrc) + rc1 * anglVel1 - rc2 * anglVel2;
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;

	// wet contact
	const double bondVolume = PI / 3. * (std::pow(radius1, 3) + std::pow(radius2, 3)); // one quarter of summarized sphere volume
	const double A = -1.1 * std::pow(bondVolume, -0.53);
	const double tempLn = std::log(bondVolume);
	const double B = (-0.34 * tempLn - 0.96) * contactAngle * contactAngle - 0.019 * tempLn + 0.48;
	const double C = 0.0042 * tempLn + 0.078;
	const CVector3 capForce = normVector * (PI *_collision->dEquivRadius * surfaceTension * (std::exp(A * bondLength + B) + C));
	const CVector3 normViscForce = normRelVel * (6 * PI * viscosity * _collision->dEquivRadius * _collision->dEquivRadius / bondLength);
	const CVector3 tangForceLiq = tangRelVel * (6 * PI * viscosity * _collision->dEquivRadius * (8. / 15. * std::log(_collision->dEquivRadius / bondLength) + 0.9588));
	const CVector3 momentLiq = rAC * tangForceLiq;

	// dry contact
	CVector3 normForceDry{ 0 }, tangForceDry{ 0 }, tangShearForceDry{ 0 }, tangDampingForceDry{ 0 }, rollingTorque1{ 0 }, rollingTorque2{ 0 }, tangOverlap{ 0 };
	if (realOverlap >= 0)
	{
		// contact radius
		const double contactRadius = std::sqrt(_collision->dEquivRadius * realOverlap);

		// normal force with damping
		const double Kn = 2 * _interactProp.dEquivYoungModulus * contactRadius;
		const double normContactForceDryLen = -2 / 3. * realOverlap * Kn;
		const double normDampingForceDryLen = -_2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * std::sqrt(Kn * _collision->dEquivMass);
		normForceDry = normVector * (normContactForceDryLen + normDampingForceDryLen);

		// rotate old tangential overlap
		CVector3 tangOverlapRot = _collision->vTangOverlap - normVector * DotProduct(normVector, _collision->vTangOverlap);
		if (tangOverlapRot.IsSignificant())
			tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
		// calculate new tangential overlap
		tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

		// tangential force with damping
		const double Kt = 8 * _interactProp.dEquivShearModulus * contactRadius;
		tangShearForceDry = tangOverlap * Kt;
		tangDampingForceDry = tangRelVel * (-_2_SQRT_5_6 * _interactProp.dAlpha * std::sqrt(Kt * _collision->dEquivMass));

		// check slipping condition and calculate total tangential force
		const double tangShearForceDryLen = tangShearForceDry.Length();
		const double frictionForceLen = _interactProp.dSlidingFriction * std::abs(normContactForceDryLen + normDampingForceDryLen);
		if (tangShearForceDryLen > frictionForceLen)
		{
			tangForceDry = tangShearForceDry * frictionForceLen / tangShearForceDryLen;
			tangOverlap  = tangForceDry / Kt;
		}
		else
			tangForceDry = tangShearForceDry + tangDampingForceDry;

		// rolling torque
		if (anglVel1.IsSignificant())
			rollingTorque1 = anglVel1 * (-_interactProp.dRollingFriction * std::abs(normContactForceDryLen) * radius1 / anglVel1.Length());
		if (anglVel2.IsSignificant())
			rollingTorque2 = anglVel2 * (-_interactProp.dRollingFriction * std::abs(normContactForceDryLen) * radius2 / anglVel2.Length());
	}

	// final forces and moments
	const CVector3 tangForce  = tangForceDry + tangForceLiq;
	const CVector3 totalForce = normForceDry + tangForce + capForce + normViscForce;
	const CVector3 moment1    = normVector * tangForce * radius1 + rollingTorque1 - momentLiq;
	const CVector3 moment2    = normVector * tangForce * radius2 + rollingTorque2 - momentLiq;

	// store results in collision
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce;
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment1;
	_collision->vResultMoment2 = moment2;
}
