/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPHertzMindlinLiquid.h"

CModelPPHertzMindlinLiquid::CModelPPHertzMindlinLiquid()
{
	m_name = "Hertz-MindlinLiquid";
	m_uniqueKey = "B18A46C2786D4D44B925AASS4D0D1008";

	AddParameter("MIN_THICKNESS", "Min. thickness of liquid [m]", 1e-5);
	AddParameter("CONTACT_ANGLE", "Contact angle [grad]", 70);
	AddParameter("SURFACE_TENSION", "Surface tension [N/m]", 0.7);
	AddParameter("DYNAMIC_VISCOSITY", "Dynamic viscosity [Pa*s]", 0.1);

	m_hasGPUSupport = false;
}

void CModelPPHertzMindlinLiquid::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dMinThickness   = m_parameters[0].value;
	const double dContactAngle   = m_parameters[1].value * PI / 180;
	const double dSurfaceTension = m_parameters[2].value;
	const double dViscosity      = m_parameters[3].value;

	double dCurrentBondLength = _pCollision->vContactVector.Length() - Particles().Radius(_iSrc) - Particles().Radius(_iDst);
	if (dCurrentBondLength < dMinThickness)
		dCurrentBondLength = dMinThickness;
	CVector3 rAC = -0.5 * _pCollision->vContactVector;

	double dRealOverlap = _pCollision->dNormalOverlap - (Particles().ContactRadius(_iSrc) - Particles().Radius(_iSrc)) - (Particles().ContactRadius(_iDst) - Particles().Radius(_iDst));

	const CVector3 vRcSrc        = _pCollision->vContactVector * ( Particles().Radius(_iSrc) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vRcDst        = _pCollision->vContactVector * (-Particles().Radius(_iDst) / (Particles().Radius(_iSrc) + Particles().Radius(_iDst)));
	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	// relative velocity (normal and tangential)
	const CVector3 vRelVel       = Particles().Vel(_iDst) - Particles().Vel(_iSrc) + vRcSrc * Particles().AnglVel(_iSrc) - vRcDst * Particles().AnglVel(_iDst);
	const double   dRelVelNormal = DotProduct(vNormalVector, vRelVel);
	const CVector3 vRelVelNormal = dRelVelNormal * vNormalVector;
	const CVector3 vRelVelTang   = vRelVel - vRelVelNormal;

	// wet contact
	double dBondVolume = PI / 3. * (std::pow(Particles().Radius(_iSrc), 3) + std::pow(Particles().Radius(_iDst), 3)); // one quarter of summarized sphere volume
	double dA = -1.1 * std::pow(dBondVolume, -0.53);
	double dTempLn = std::log(dBondVolume);
	double dB = (-0.34 * dTempLn - 0.96) * dContactAngle * dContactAngle - 0.019 * dTempLn + 0.48;
	double dC = 0.0042 * dTempLn + 0.078;
	CVector3 dCapForce = vNormalVector * (PI *_pCollision->dEquivRadius * dSurfaceTension * (std::exp(dA * dCurrentBondLength + dB) + dC));
	CVector3 dViscForceNormal = vRelVelNormal * (6 * PI * dViscosity * _pCollision->dEquivRadius * _pCollision->dEquivRadius / dCurrentBondLength);
	CVector3 vTangForceLiquid = vRelVelTang * (6 * PI * dViscosity * _pCollision->dEquivRadius * (8. / 15. * std::log(_pCollision->dEquivRadius / (dCurrentBondLength)) + 0.9588));
	CVector3 vMomentLiquid = rAC * vTangForceLiquid;

	// dry contact
	CVector3 vNormalForce(0), vTangForceDry(0), vDampingTangForceDry(0);
	CVector3 vRollingTorque1(0), vRollingTorque2(0);
	if (dRealOverlap < 0)
		_pCollision->vTangOverlap.Init(0);
	else // dry force
	{
		// normal force with damping
		const double Kn = 2 * _interactProp.dEquivYoungModulus * std::sqrt(_pCollision->dEquivRadius*dRealOverlap);
		const double dDampingNormalForceDry = -1.8257 * _interactProp.dAlpha * dRelVelNormal * std::sqrt(Kn * _pCollision->dEquivMass);
		const double dNormalForceDry = -1 * 2 / 3. * dRealOverlap*Kn;

		// increment of tangential force with damping
		double Kt = 8 * _interactProp.dEquivShearModulus * std::sqrt(_pCollision->dEquivRadius * dRealOverlap);
		vDampingTangForceDry = vRelVelTang * (-1.8257 * _interactProp.dAlpha * std::sqrt(Kt * _pCollision->dEquivMass));

		// rotate old tangential force
		CVector3 vTangOverlap = _pCollision->vTangOverlap - vNormalVector * DotProduct(vNormalVector, _pCollision->vTangOverlap);
		if (vTangOverlap.IsSignificant())
			vTangOverlap = vTangOverlap * _pCollision->vTangOverlap.Length() / vTangOverlap.Length();
		_pCollision->vTangOverlap = vTangOverlap + vRelVelTang * _timeStep;

		vTangForceDry = _pCollision->vTangOverlap * Kt;

		// check slipping condition
		double dNewTangForce = vTangForceDry.Length();
		if (dNewTangForce > _interactProp.dSlidingFriction * std::abs(dNormalForceDry))
		{
			vTangForceDry *= _interactProp.dSlidingFriction * std::abs(dNormalForceDry) / dNewTangForce;
			_pCollision->vTangOverlap = vTangForceDry / Kt;
		}
		else
			vTangForceDry += vDampingTangForceDry;

		// calculate rolling friction
		if (Particles().AnglVel(_iSrc).IsSignificant()) // if it is not zero, but small enough, its Length() can turn into zero and division fails
			vRollingTorque1 = Particles().AnglVel(_iSrc) * (-_interactProp.dRollingFriction * std::abs(dNormalForceDry) * Particles().Radius(_iSrc) / Particles().AnglVel(_iSrc).Length());
		if (Particles().AnglVel(_iDst).IsSignificant()) // if it is not zero, but small enough, its Length() can turn into zero and division fails
			vRollingTorque2 = Particles().AnglVel(_iDst) * (-_interactProp.dRollingFriction * std::abs(dNormalForceDry) * Particles().Radius(_iDst) / Particles().AnglVel(_iDst).Length());

		vNormalForce = vNormalVector * (dNormalForceDry + dDampingNormalForceDry);
	}
	// save tangential force
	_pCollision->vTangForce = vTangForceDry + vTangForceLiquid;

	_pCollision->vTotalForce    = vNormalForce + _pCollision->vTangForce + dCapForce + dViscForceNormal;
	_pCollision->vResultMoment1 = vNormalVector * _pCollision->vTangForce*Particles().Radius(_iSrc) + vRollingTorque1 - vMomentLiquid;
	_pCollision->vResultMoment2 = vNormalVector * _pCollision->vTangForce*Particles().Radius(_iDst) + vRollingTorque2 - vMomentLiquid;
}
