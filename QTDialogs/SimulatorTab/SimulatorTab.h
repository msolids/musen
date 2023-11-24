/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "SimulatorManager.h"
#include "ModelsConfiguratorTab.h"
#include "ui_SimulatorTab.h"
#include <QSettings>
#include <QElapsedTimer>
#include <QTimer>
#include <QDateTime>

class CSimulatorThread : public QObject
{
	Q_OBJECT
public:
	CBaseSimulator* m_pSimulator;
	public slots:
	void StartSimulation();
	void StopSimulation();
	void PauseSimulation();
signals:
	void finished();
};

class CSimulatorTab : public CMusenDialog
{
	Q_OBJECT
public:
	bool m_bShowSimDomain;

private:
	enum EStatTable : unsigned
	{
		SIM_TIME = 0,
		SIM_TIME_STEP = 1,
		MAX_PART_VELO = 2,
		NUM_BROKEN_S_BONDS = 3,
		NUM_BROKEN_L_BONDS = 4,
		NUM_GENERATED = 5,
		NUM_INACTIVE = 6,
		SIM_STARTED = 7,
		SIM_FINISHED = 8,
		SIM_LEFT = 9,
		SIM_ELAPSED = 10
	};
	Ui::simulatorTab ui;
	QSettings* m_pSettings;

	CSimulatorManager* m_pSimulatorManager;
	QTimer m_UpdateTimer;	// timer which is used to update statistic
	CSimulatorThread* m_pDEMThreadNew;
	QThread* m_pQTThread;
	bool m_bSimulationStarted;

public:
	CSimulatorTab(CSimulatorManager* _pSimManager, QSettings* _pSettings, QWidget *parent = 0);

private:
	void InitializeConnections() const;
	void UpdateSimulationDomain();

private slots:
	void SetParameters();
	void StartButtonClicked();
	void StartSimulation();
	void ResumeSimulation();
	void StopSimulation();
	void PauseSimulation();
	void SimulationFinished();
	void UpdateRayleighTime();
	void UpdateSimulationStatistics() const;
	void UpdateWholeView() override;
	void UpdateModelsView() const;
	void UpdateCollisionsFlag() const;
	void ConfigureModels();
	void SetSelectiveSaving();
	void ConfigureSelectiveSaving();

	// simulation volume
	void SimDomainChanged();
	void RecalculateSimulationDomain();
	void UpdateGUI(ERunningStatus _status);

signals:
	void SimulatorStatusChanged(ERunningStatus _nStatus);
	void NumberOfTimePointsChanged();
};