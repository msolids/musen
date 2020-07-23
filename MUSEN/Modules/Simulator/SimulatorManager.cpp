/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SimulatorManager.h"

CSimulatorManager::CSimulatorManager()
{
	m_pSimulator = new CCPUSimulator();
}

CSimulatorManager::~CSimulatorManager()
{
	delete m_pSimulator;
}

void CSimulatorManager::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	m_pSystemStructure = _pSystemStructure;
	m_pSimulator->SetSystemStructure(_pSystemStructure);
}

void CSimulatorManager::LoadConfiguration()
{
	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!protoMessage.has_simulator()) return;
	const ProtoModuleSimulator& sim = protoMessage.simulator();
	CreateSimulator(static_cast<ESimulatorType>(sim.simulator_type()));
	m_pSimulator->LoadConfiguration();
}

void CSimulatorManager::SaveConfiguration()
{
	m_pSimulator->SaveConfiguration();
}

void CSimulatorManager::SetSimulatorType(const ESimulatorType& _type)
{
	CreateSimulator(_type);
}

CBaseSimulator* CSimulatorManager::GetSimulatorPtr() const
{
	return m_pSimulator;
}

void CSimulatorManager::CreateSimulator(const ESimulatorType& _type)
{
	if (_type == m_pSimulator->GetType()) return; // the same type is requested

	CBaseSimulator* pNewSimulator = nullptr;

	switch (_type)
	{
	case ESimulatorType::BASE:
		break;
	case ESimulatorType::CPU:
		pNewSimulator = new CCPUSimulator(*m_pSimulator);
		break;
	case ESimulatorType::GPU:
		pNewSimulator = new CGPUSimulator(*m_pSimulator);
		break;
	}

	if (pNewSimulator)
	{
		pNewSimulator->SetType(_type);

		delete m_pSimulator;
		m_pSimulator = pNewSimulator;
	}
}
