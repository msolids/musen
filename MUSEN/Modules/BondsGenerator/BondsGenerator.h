/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "SystemStructure.h"
#include "MUSENComponent.h"
#include "GeneratorComponent.h"

// Information about generating bonds.
struct SBondClass
{
	bool isActive{};			// Activity of the bond's class.
	std::string name;			// Name of the bond's class.
	double minDistance{ 0.0 };	// Minimum distance between surfaces of particles to be connected.
	double maxDistance{ 1e-3 };	// Maximum distance between surfaces of particles to be connected.
	double diameter{ 5e-4 };	// Diameter of the bond.
	bool isOverlayAllowed{};	// Allow to define multiple bonds between the same particles.
	std::string compoundKey;	// Compound of the bond.

	bool isCompoundSpecific{};	// Connect only particles of specific compounds.
	std::pair<std::vector<std::string>, std::vector<std::string>> compoundsLists; // List of allowed compounds pairs that may be connected if isCompoundSpecific is set.

	unsigned generatedBonds{};	// Number of currently generated bonds.
	double completeness{};		// Current completeness in percent.
};

class CBondsGenerator : public CMusenComponent, public IGenerator<SBondClass>
{
public:
	bool IsDataCorrect() const;			// Checks correctness of data in all generators.

	void StartGeneration();				// Starts generation of bonds.

	void LoadConfiguration() override;	// Uses the same file as system structure to load configuration.
	void SaveConfiguration() override;	// Uses the same file as system structure to store configuration.
};
