/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <GeneralMUSENDialog.h>
#include "ui_ClearSpecificTPTab.h"
#include "ClearSpecificTimePoints.h"
#include <QTimer>

// Thread
class CClearSpecificThread : public QObject
{
	Q_OBJECT

public:
	CClearSpecificTimePoints* m_pClearSpecificTimePoints;

public:
	CClearSpecificThread(CClearSpecificTimePoints* _pClearSpecificTimePoints, QObject *parent = 0);
	~CClearSpecificThread();

public slots:
	void StartRemoving();

signals:
	void finished();
};

class CClearSpecificTPTab : public CMusenDialog
{
	Q_OBJECT

private:
	double dTimeStep;					   // time step for clearing
	std::vector<double> m_vAllTimePoints;  // all time points

	std::vector<size_t> m_vIndexesOfSelectedTPs;  // indexes of selected time points
	int m_nExpectedTimePoints;					  // number of expected time points after removing

	CClearSpecificThread*	  m_ClearSpecificThread;
	QThread*                  m_pQTThread;
	QTimer				      m_UpdateTimer;
	CClearSpecificTimePoints* m_pClearSpecificTimePoints;

public:
	CClearSpecificTPTab(QWidget *parent = Q_NULLPTR);
	~CClearSpecificTPTab();

private:
	Ui::CClearSpecificTPTab ui;

	// Connects qt objects to slots
	void InitializeConnections();
	// Updates progress bar and status description
	void UpdateProgressInfo();
	// Hot keys handling
	void keyPressEvent(QKeyEvent* event);
	// Fills list of time points and other fields
	void UpdateWholeView();
	// Enable/Disable section of desired time step
	void SetDesiredTimeStepSectionEnabled(bool _bEnabled);

private slots:
	// Calculates number of selected time points
	void ChangeSelection();
	// Determines indexes of time points according to time step
	void SelectTimeStep();
	// Determines indexes of each second time point
	void SelectEachSecondTimePoint();
	// Checks input parameters and calls the main Clear function in separate thread
	void RemoveButtonPressed();
	// Is used when clearing is finished
	void RemovingFinished();
};
