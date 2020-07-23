/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPWHertzMindlinLiquid.h"

CModelPWHertzMindlinLiquid::CModelPWHertzMindlinLiquid()
{
	m_name = "Hertz-MindlinLiquid";
	m_uniqueKey = "906949ACFFAE4B8C8B1B62209930EA6D";
	//m_sHelpFileName = "/Contact Models/HertzMindlinLiquid.pdf";

	AddParameter("MIN_THICKNESS", "Min. thickness of liquid [m]", 1e-5);
	AddParameter("CONTACT_ANGLE", "Contact angle [grad]", 70);
	AddParameter("SURFACE_TENSION", "Surface tension [N/m]", 0.7);
	AddParameter("DYNAMIC_VISCOSITY", "Dynamic viscosity [Pa*s]", 0.1);

	m_hasGPUSupport = true;
}

void CModelPWHertzMindlinLiquid::CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dMinThickness = m_parameters[0].value;
	const double dContactAngle = m_parameters[1].value * PI / 180;
	const double dSurfaceTension = m_parameters[2].value;
	const double dViscosity = m_parameters[3].value;

	const CVector3 vRc          = CPU_GET_VIRTUAL_COORDINATE(Particles().Coord(_iPart)) - _pCollision->vContactVector;
	const double   dRc          = vRc.Length();
	const CVector3 nRc          = vRc / dRc; // = vRc.Normalized()
	const CVector3 vVelDueToRot = !Walls().RotVel(_iWall).IsZero() ? (_pCollision->vContactVector - Walls().RotCenter(_iWall)) * Walls().RotVel(_iWall) : CVector3{ 0 };

	double dBondLength = dRc;
	if (dBondLength < dMinThickness)
		dBondLength = dMinThickness;

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       =  Particles().Vel(_iPart) - Walls().Vel(_iWall) + vVelDueToRot + nRc * Particles().AnglVel(_iPart) * Particles().Radius(_iPart);
	const double   dRelVelNormal = DotProduct(Walls().NormalVector(_iWall), vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * Walls().NormalVector(_iWall);
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// wet contact
	const double dBondVolume = PI / 3.0 * std::pow(Particles().Radius(_iPart), 3); // one quarter of summarized sphere volume
	const double dA = -1.1 * std::pow(dBondVolume, -0.53);
	const double dTempLn = std::log(dBondVolume);
	const double dB = (-0.34 * dTempLn - 0.96) * dContactAngle * dContactAngle - 0.019 * dTempLn + 0.48;
	const double dC = 0.0042 * dTempLn + 0.078;
	const CVector3 dCapForce = Walls().NormalVector(_iWall) * PI * Particles().Radius(_iPart) * dSurfaceTension * (std::exp(dA * dBondLength + dB) + dC);
	const CVector3 dViscForceNormal = vRelVelNormal * -1 * (6 * PI * dViscosity * Particles().Radius(_iPart) * Particles().Radius(_iPart) / dBondLength);
	const CVector3 vTangForceLiquid = vRelVelTang * (6 * PI * dViscosity * Particles().Radius(_iPart) * (8.0 / 15.0 * std::log(Particles().Radius(_iPart) / dBondLength) + 0.9588));
	const CVector3 vMomentLiquid = vRc * vTangForceLiquid;

	// normal and Tangential overlaps
	const double dNormalOverlap =  Particles().Radius(_iPart) - dRc;
	CVector3 vNormalForce(0), vTangForceDry(0), vDampingTangForceDry(0), vRollingTorqueDry(0);
	if (dNormalOverlap < 0)
	{
		_pCollision->vTangOverlap.Init(0);
	}
	else // dry contact
	{
		// normal force with damping
		const double Kn = 2 * _interactProp.dEquivYoungModulus * std::sqrt(Particles().Radius(_iPart) * dNormalOverlap);
		const double dDampingNormalForceDry = 1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn *  Particles().Mass(_iPart));
		const double dNormalForceDry =  2 / 3. * dNormalOverlap * Kn * std::abs(DotProduct(nRc, Walls().NormalVector(_iWall)));

		// increment of tangential force with damping
		const double Kt = 8 * _interactProp.dEquivShearModulus * std::sqrt( Particles().Radius(_iPart) * dNormalOverlap);
		vDampingTangForceDry = vRelVelTang * (1.8257 * _interactProp.dAlpha * std::sqrt(Kt *  Particles().Mass(_iPart)));

		// rotate old tangential force
		CVector3 vTangOverlap = _pCollision->vTangOverlap - Walls().NormalVector(_iWall) * DotProduct(Walls().NormalVector(_iWall), _pCollision->vTangOverlap);
		if (vTangOverlap.IsSignificant())
			vTangOverlap = vTangOverlap * _pCollision->vTangOverlap.Length() / vTangOverlap.Length();

		_pCollision->vTangOverlap = vTangOverlap + vRelVelTang * _timeStep;
		vTangForceDry = _pCollision->vTangOverlap * -Kt;
		// check slipping condition
		const double dNewTangForce = vTangForceDry.Length();
		if (dNewTangForce > _interactProp.dSlidingFriction * std::abs(dNormalForceDry))
		{
			vTangForceDry = vTangForceDry * (_interactProp.dSlidingFriction * std::abs(dNormalForceDry) / dNewTangForce);
			_pCollision->vTangOverlap = vTangForceDry / -Kt;
		}

		// calculate rolling friction
		if (Particles().AnglVel(_iPart).IsSignificant())
			vRollingTorqueDry =  Particles().AnglVel(_iPart) * (-_interactProp.dRollingFriction * std::abs(dNormalForceDry) *  Particles().Radius(_iPart) /  Particles().AnglVel(_iPart).Length());

		vNormalForce = Walls().NormalVector(_iWall) * (dNormalForceDry + dDampingNormalForceDry);
	}

	// save old tangential force
	_pCollision->vTangForce = vTangForceDry + vDampingTangForceDry + vTangForceLiquid;

	// add result to the arrays
	_pCollision->vTotalForce = vNormalForce + _pCollision->vTangForce + dCapForce + dViscForceNormal;
	_pCollision->vResultMoment1 = Walls().NormalVector(_iWall) * _pCollision->vTangForce * - Particles().Radius(_iPart) + vRollingTorqueDry - vMomentLiquid;
}
