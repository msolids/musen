/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "PackageGeneratorHelper.h"
#include "SceneTypesGPU.h"
#include <thrust/device_vector.h>

class CPackageGeneratorHelperGPU : public IPackageGeneratorHelper
{
	size_t m_number{};									// Number of particles.
	SGPUParticles* m_particles;							// Pointer to particles.
	thrust::device_vector<CVector3>* m_oldVels{};		// Old velocities.
	thrust::device_vector<CVector3>* m_deltaCoord{};	// Temporary variable, needed for speed-up.
	thrust::device_vector<double>* m_relVels{};			// Temporary variable, needed for speed-up.

public:
	CPackageGeneratorHelperGPU(SGPUParticles* _particles);
	CPackageGeneratorHelperGPU(const CPackageGeneratorHelperGPU& _other)                = delete;
	CPackageGeneratorHelperGPU(CPackageGeneratorHelperGPU&& _other) noexcept            = delete;
	CPackageGeneratorHelperGPU& operator=(const CPackageGeneratorHelperGPU& _other)     = delete;
	CPackageGeneratorHelperGPU& operator=(CPackageGeneratorHelperGPU&& _other) noexcept = delete;
	~CPackageGeneratorHelperGPU();

	void LimitVelocities() const override;
	void ScaleVelocitiesToRadius(double _minRadius) const override;
	double MaxRelativeVelocity() const override;
	void ResetMovement() override;
	void SaveVelocities() override;
};
