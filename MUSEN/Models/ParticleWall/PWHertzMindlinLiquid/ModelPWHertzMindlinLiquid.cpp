/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHertzMindlinLiquid.h"

CModelPWHertzMindlinLiquid::CModelPWHertzMindlinLiquid()
{
	m_name          = "Hertz-MindlinLiquid";
	m_uniqueKey     = "906949ACFFAE4B8C8B1B62209930EA6D";
	m_hasGPUSupport = true;

	/* 0*/ AddParameter("MIN_THICKNESS"    , "Min. thickness of liquid [m]", 1e-5);
	/* 1*/ AddParameter("CONTACT_ANGLE"    , "Contact angle [grad]"        , 70  );
	/* 2*/ AddParameter("SURFACE_TENSION"  , "Surface tension [N/m]"       , 0.7 );
	/* 3*/ AddParameter("DYNAMIC_VISCOSITY", "Dynamic viscosity [Pa*s]"    , 0.1 );
}

void CModelPWHertzMindlinLiquid::CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double minThickness   = m_parameters[0].value;
	const double contactAngle   = m_parameters[1].value * PI / 180;
	const double surfaceTension = m_parameters[2].value;
	const double viscosity      = m_parameters[3].value;

	const double   partRadius  = Particles().Radius(_iPart);
	const CVector3 partAnglVel = Particles().AnglVel(_iPart);
	const CVector3 normVector  = Walls().NormalVector(_iWall);

	const CVector3 rc     = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _collision->vContactVector;
	const double   rcLen  = rc.Length();
	const CVector3 rcNorm = rc / rcLen;

	const double bondLength = std::max(rcLen, minThickness);

	// normal overlap
	const double normOverlap =  partRadius - rcLen;

	// normal and tangential relative velocity
	const CVector3 rotVel        = !Walls().RotVel(_iWall).IsZero() ? (_collision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };
	const CVector3 relVel        = Particles().Vel(_iPart) - Walls().Vel(_iWall) + rotVel + rcNorm * partAnglVel * partRadius;
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;

	// wet contact
	const double bondVolume = PI / 3.0 * std::pow(partRadius, 3); // one quarter of summarized sphere volume
	const double A = -1.1 * std::pow(bondVolume, -0.53);
	const double tempLn = std::log(bondVolume);
	const double B = (-0.34 * tempLn - 0.96) * contactAngle * contactAngle - 0.019 * tempLn + 0.48;
	const double C = 0.0042 * tempLn + 0.078;
	const CVector3 capForce = normVector * PI * partRadius * surfaceTension * (std::exp(A * bondLength + B) + C);
	const CVector3 normViscForce = normRelVel * -1 * (6 * PI * viscosity * partRadius * partRadius / bondLength);
	const CVector3 tangForceLiq = tangRelVel * (6 * PI * viscosity * partRadius * (8.0 / 15.0 * std::log(partRadius / bondLength) + 0.9588));
	const CVector3 momentLiq = rc * tangForceLiq;

	// dry contact
	CVector3 normForceDry{ 0 }, tangForceDry{ 0 }, tangShearForceDry{ 0 }, tangDampingForceDry{ 0 }, rollingTorqueDry{ 0 }, tangOverlap{ 0 };
	if (normOverlap >= 0)
	{
		// contact radius
		const double contactRadius = std::sqrt(partRadius * normOverlap);

		// normal force with damping
		const double Kn = 2 * _interactProp.dEquivYoungModulus * contactRadius;
		const double normContactForceDryLen = 2 / 3. * normOverlap * Kn * std::abs(DotProduct(rcNorm, normVector));
		const double normDampingForceDryLen = _2_SQRT_5_6 * _interactProp.dAlpha * normRelVelLen * std::sqrt(Kn * Particles().Mass(_iPart));
		normForceDry = normVector * (normContactForceDryLen + normDampingForceDryLen);

		// rotate old tangential overlap
		CVector3 tangOverlapRot = _collision->vTangOverlap - normVector * DotProduct(normVector, _collision->vTangOverlap);
		if (tangOverlapRot.IsSignificant())
			tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
		// calculate new tangential overlap
		tangOverlap = tangOverlapRot + tangRelVel * _timeStep;

		// tangential force with damping
		const double Kt = 8 * _interactProp.dEquivShearModulus * contactRadius;
		tangShearForceDry = -Kt * tangOverlap;
		tangDampingForceDry = tangRelVel * (_2_SQRT_5_6 * _interactProp.dAlpha * std::sqrt(Kt * Particles().Mass(_iPart)));

		// check slipping condition and calculate total tangential force
		const double tangShearForceDryLen = tangShearForceDry.Length();
		const double frictionForceLen = _interactProp.dSlidingFriction * std::abs(normContactForceDryLen + normDampingForceDryLen);
		if (tangShearForceDryLen > frictionForceLen)
		{
			tangForceDry = tangShearForceDry * frictionForceLen / tangShearForceDryLen;
			tangOverlap  = tangForceDry / -Kt;
		}
		else
			tangForceDry = tangShearForceDry + tangDampingForceDry;

		// rolling torque
		rollingTorqueDry = partAnglVel.IsSignificant() ? partAnglVel * (-_interactProp.dRollingFriction * std::abs(normContactForceDryLen) * partRadius / partAnglVel.Length()) : CVector3{ 0 };
	}

	// final forces and moments
	const CVector3 tangForce  = tangForceDry + tangForceLiq;
	const CVector3 totalForce = normForceDry + tangForce + capForce + normViscForce;
	const CVector3 moment1    = normVector * tangForce * -partRadius + rollingTorqueDry - momentLiq;

	// store results in collision
	_collision->vTangOverlap   = tangOverlap;
	_collision->vTangForce     = tangForce;
	_collision->vTotalForce    = totalForce;
	_collision->vResultMoment1 = moment1;
}

void CModelPWHertzMindlinLiquid::ConsolidatePart(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const
{
	_particles.Force(_iPart) += _collision->vTotalForce;
	_particles.Moment(_iPart) += _collision->vResultMoment1;
}

void CModelPWHertzMindlinLiquid::ConsolidateWall(double _time, double _timeStep, size_t _iWall, SWallStruct& _walls, const SCollision* _collision) const
{
	_walls.Force(_iWall) -= _collision->vTotalForce;
}