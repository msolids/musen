/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "AbstractDEMModel.h"

class CModelEF : public CExternalForceModel
{
public:
	//////////////////////////////////////////////////////////////////////////
	/// Constructor.
	CModelEF();

	//////////////////////////////////////////////////////////////////////////
	/// CPU implementation.

	/// Is called each time step before real calculations. Can be removed if not used.
	void PrecalculateEF(double _time, double _timeStep, SParticleStruct* _particles) override;
	/// The model itself.
	void CalculateEF(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const override;
	//////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////
	/// GPU implementation. Both functions can be removed if no GPU implementation provided.

	/// Set model parameters to GPU. Should not be changed.
	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	/// Invokes the GPU-version of the model. Can be adjusted.
	void CalculateEFGPU(double _time, double _timeStep, SGPUParticles& _particles) override;
	//////////////////////////////////////////////////////////////////////////
};

