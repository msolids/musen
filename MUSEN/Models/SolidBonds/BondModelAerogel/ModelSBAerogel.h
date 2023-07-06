/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "AbstractDEMModel.h"

class CModelSBAerogel : public CSolidBondModel
{
public:
	CModelSBAerogel();
	void CalculateSBForce(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const override;
	void ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const override;

	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	void CalculateSBForceGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds) override;
};