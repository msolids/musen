/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "PackageGeneratorHelper.h"
#include "SceneTypes.h"

class CPackageGeneratorHelperCPU : public IPackageGeneratorHelper
{
	size_t m_number{};					// Number of particles.
	SParticleStruct* m_particles;		// Pointer to particles.
	std::vector<CVector3> m_oldVels;	// Old velocities.

public:
	CPackageGeneratorHelperCPU(SParticleStruct* _particles);

	void LimitVelocities() const override;
	void ScaleVelocitiesToRadius(double _minRadius) const override;
	double MaxRelativeVelocity() const override;
	void ResetMovement() override;
	void SaveVelocities() override;
};


