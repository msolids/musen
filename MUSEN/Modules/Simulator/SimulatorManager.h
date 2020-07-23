/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "CPUSimulator.h"
#include "GPUSimulator.h"

class CSimulatorManager : public CMusenComponent
{
private:
	CBaseSimulator* m_pSimulator;

public:
	CSimulatorManager();
	~CSimulatorManager();

	void SetSystemStructure(CSystemStructure* _pSystemStructure) override;

	void LoadConfiguration() override; // Uses the same file as system structure to load configuration.
	void SaveConfiguration() override; // Uses the same file as system structure to store configuration.

	void SetSimulatorType(const ESimulatorType& _type); // Replaces simulator with a new one of specified type and copies all parameters from the current simulator.

	CBaseSimulator* GetSimulatorPtr() const;	// Returns pointer to a currently selected simulator.

private:
	void CreateSimulator(const ESimulatorType& _type);
};

