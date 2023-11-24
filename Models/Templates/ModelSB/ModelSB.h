/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "AbstractDEMModel.h"

class CModelSB : public CSolidBondModel
{
public:
	//////////////////////////////////////////////////////////////////////////
	/// Constructor.
	CModelSB();

	//////////////////////////////////////////////////////////////////////////
	/// CPU implementation.

	/// Is called each time step before real calculations. Can be removed if not used.
	void PrecalculateSB(double _time, double _timeStep, SParticleStruct* _particles, SSolidBondStruct* _bonds) override;
	/// The model itself.
	void CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const override;
	/// Apply calculated bond to a particle.
	void ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const override;
	//////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////
	/// GPU implementation. Both functions can be removed if no GPU implementation provided.

	/// Set model parameters to GPU. Should not be changed.
	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	/// Invokes the GPU-version of the model. Can be adjusted.
	void CalculateSBGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds) override;
	//////////////////////////////////////////////////////////////////////////
};

