/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "AbstractDEMModel.h"

class CModelPP : public CParticleParticleModel
{
public:
	//////////////////////////////////////////////////////////////////////////
	/// Constructor.
	CModelPP();

	//////////////////////////////////////////////////////////////////////////
	/// CPU implementation.

	/// Is called each time step before real calculations. Can be removed if not used.
	void PrecalculatePP(double _time, double _timeStep, SParticleStruct* _particles) override;
	/// The model itself.
	void CalculatePP(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const override;
	/// Apply calculated collision to a particle.
	void ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const override;
	/// Apply calculated collision to a particle.
	void ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const override;
	//////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////
	/// GPU implementation. Both functions can be removed if no GPU implementation provided.

	/// Set model parameters to GPU. Should not be changed.
	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	/// Invokes the GPU-version of the model. Can be adjusted.
	void CalculatePPGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions) override;
	//////////////////////////////////////////////////////////////////////////
};

