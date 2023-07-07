/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "AbstractDEMModel.h"

class CModelLBCapilarViscous : public CLiquidBondModel
{
public:
	CModelLBCapilarViscous();

	void CalculateLB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SLiquidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const override;
	void ConsolidatePart(double _time, double _timeStep, size_t _iBond, SParticleStruct& _particles) const override;
};