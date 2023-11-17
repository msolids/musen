/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "PackageGenerator.h"

// Used to filter out unnecessary particles.
class CParticleFilter
{
	CSystemStructure& m_systemStructure;					// Pointer to original system structure.
	const SPackage& m_generator;							// Pointer to a package generator.

	CInsideVolumeChecker* m_volumeChecker{ nullptr };		// Checker to verify objects are inside the generation volume.
	std::vector<CInsideVolumeChecker*> m_geometryCheckers;	// Checkers to verify new objects are not situated in any real volume.

public:
	CParticleFilter(CSystemStructure& _systemStructure, const SPackage& _generator);
	~CParticleFilter();

	std::vector<size_t> Filter(const std::vector<CVector3>& _coords, const std::vector<double>& _radii);
};

