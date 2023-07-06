/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "AbstractDEMModel.h"

class CModelPWHertzMindlinLiquid : public CParticleWallModel
{
public:
	CModelPWHertzMindlinLiquid();
	void CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _collision) const override;
	void ConsolidatePart(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const override;
	void ConsolidateWall(double _time, double _timeStep, size_t _iWall, SWallStruct& _walls, const SCollision* _collision) const override;

	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	void CalculatePWForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions) override;
};