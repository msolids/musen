/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
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

/// Factory to create a CPU package generator helper.
/// @param _particles Pointer to particles on CPU.
/// @return Pointer to the created CPU package generator helper.
IPackageGeneratorHelper* CreatePackageGeneratorHelperCPU(SParticleStruct* _particles);
