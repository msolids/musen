/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelPPSintering.h"

CModelPPSintering::CModelPPSintering()
{
	m_name          = "Sintering model";
	m_uniqueKey     = "A41AB40D3A074FE9AAE0B0ADB27ABFBA";
	m_helpFileName  = "/Contact Models/Sintering.pdf";
	m_hasGPUSupport = true;

	/* 0*/ AddParameter("DIFFUSION"        , "Diffusion parameter [??]"                        , 1.28e-34);
	/* 1*/ AddParameter("VISCOUS_PARAMETER", "Viscous parameter for tangential force (eta) [-]", 0.01);
}

void CModelPPSintering::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double diffusionParam = m_parameters[0].value;
	const double viscousParam   = m_parameters[1].value;

	const CVector3 normVector = _collision->vContactVector.Normalized();

	// normal and tangential relative velocity
	const CVector3 relVel     =  Particles().Vel(_iSrc) -  Particles().Vel(_iDst);
	const CVector3 normRelVel = normVector * DotProduct(normVector, relVel);
	const CVector3 tangRelVel = relVel - normRelVel;

	// Bouvard and McMeeking's model
	const double squaredContactRadius = 4 * _collision->dEquivRadius * _collision->dNormalOverlap;

	// forces
	const CVector3 sinteringForce = normVector * 1.125 * PI * 2 * _collision->dEquivRadius * _interactProp.dEquivSurfaceEnergy;
	const CVector3 viscousForce   = normRelVel * (-PI * std::pow(squaredContactRadius, 2) / 8 / diffusionParam);
	const CVector3 tangForce      = tangRelVel * (-viscousParam * PI * squaredContactRadius * std::pow(2 * _collision->dEquivRadius, 2) / 8 / diffusionParam);
	const CVector3 totalForce     = sinteringForce + viscousForce + tangForce;

	// store results in collision
	_collision->vTotalForce = totalForce;
}
