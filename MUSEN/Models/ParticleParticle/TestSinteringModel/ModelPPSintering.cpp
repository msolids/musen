/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSintering.h"

CModelPPSintering::CModelPPSintering()
{
	m_name = "Sintering model";
	m_uniqueKey = "A41AB40D3A074FE9AAE0B0ADB27ABFBA";
	m_helpFileName = "/Contact Models/Sintering.pdf";

	AddParameter("DIFFUSION", "Diffusion parameter [??]", 1.28e-34);
	AddParameter("VISCOUS_PARAMETER", "Viscous parameter for tangential force (eta) [-]", 0.01);

	m_hasGPUSupport = true;
}

void CModelPPSintering::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const
{
	const double dDiffusionParameter = m_parameters[0].value;
	const double dViscousParameter   = m_parameters[1].value;

	const CVector3 vNormalVector = _pCollision->vContactVector.Normalized();

	//obtain velocities
	const CVector3 vRelVel           =  Particles().Vel(_iSrc) -  Particles().Vel(_iDst);
	const CVector3 vRelVelNormal     = vNormalVector * DotProduct(vNormalVector, vRelVel);
	const CVector3 vRelVelTangential = vRelVel - vRelVelNormal;

	//Bouvard and McMeeking's model
	const double dSquaredContactRadius = 4 * _pCollision->dEquivRadius * _pCollision->dNormalOverlap;

	// calculate forces
	const CVector3 vSinteringForce = vNormalVector * 1.125 * PI * 2 * _pCollision->dEquivRadius * _interactProp.dEquivSurfaceEnergy;
	const CVector3 vViscousForce = vRelVelNormal * (-PI * std::pow(dSquaredContactRadius, 2) / 8 / dDiffusionParameter);
	const CVector3 vTangentialForce = vRelVelTangential * (-dViscousParameter * PI * dSquaredContactRadius * std::pow(2 * _pCollision->dEquivRadius, 2) / 8 / dDiffusionParameter);
	const CVector3 vResultForce = vSinteringForce + vViscousForce + vTangentialForce;

	// first particle
	_pCollision->vTotalForce = vResultForce;
}
