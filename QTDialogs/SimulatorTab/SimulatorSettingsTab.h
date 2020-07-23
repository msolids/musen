/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_SimulatorSettingsTab.h"
#include "SimulatorManager.h"
#include "GeneralMUSENDialog.h"

class CSimulatorSettingsTab : public CMusenDialog
{
	Q_OBJECT

	Ui::CSimulatorSettingsTab ui;
	CSimulatorManager* m_pSimulatorManager;
	bool m_bThreadPoolChanged{ false };

public:
	CSimulatorSettingsTab(CSimulatorManager* _pSimulatorManager, QWidget *parent = Q_NULLPTR);

	void UpdateWholeView() override;

private:
	void UpdateCPUList() const;
	void SetCPUList();

private slots:
	void ThreadPoolChanged();
	void AcceptChanges();
};
