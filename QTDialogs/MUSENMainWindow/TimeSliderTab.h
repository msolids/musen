/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_TimeSliderTab.h"
#include "GeneralMUSENDialog.h"
#include "UnitConvertor.h"

class CTimeSliderTab : public CMusenDialog
{
	Q_OBJECT
public:
	Ui::timeSliderTab ui;

private:
	void SetTime( double _dTime );

public slots:
	void UpdateWholeView();
	void SetTimeSliderEnabled();

private slots:
	void ChangeCurrentTime(); // change time via slider
	void SetSpecificTime();  // change time via set button
	void GoToLastTimePoint();
	void GoToFirstTimePoint();
	void GoToNextTimePoint();
	void GoToPreviousTimePoint();

signals:
	void NewTimeSelected();

public:
	CTimeSliderTab(  QWidget *parent = 0 );

	double GetCurrentTime(){ return m_dCurrentTime; }
};

