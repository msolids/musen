/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_SelectiveSavingTab.h"
#include "SimulatorManager.h"

class CSelectiveSavingTab : public CMusenDialog
{
	Q_OBJECT

private:
	Ui::CSelectiveSavingTab ui;

	CSimulatorManager* m_pSimulatorManager;
	SSelectiveSavingFlags m_SSelectiveSavingFlags;

public:
	CSelectiveSavingTab(CSimulatorManager* _pSimManager, QWidget* parent = Q_NULLPTR);

private:
	void InitializeConnections() const;
	void UpdateWholeView() override;

private slots:
	void SetParameters();
};