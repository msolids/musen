/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "AbstractDEMModel.h"

class CModelPPSimpleViscoElastic : public CParticleParticleModel
{
public:
	CModelPPSimpleViscoElastic();
	void CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const override;
	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	void CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions) override;
};