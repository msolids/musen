/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "AbstractDEMModel.h"

class CModelPPHertzMindlinLiquid : public CParticleParticleModel
{
public:
	CModelPPHertzMindlinLiquid();
	void CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _pCollision) const override;
};