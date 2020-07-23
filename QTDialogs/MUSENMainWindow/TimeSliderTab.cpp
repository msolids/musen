/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TimeSliderTab.h"
#include "qtOperations.h"

CTimeSliderTab::CTimeSliderTab( QWidget *parent ) :CMusenDialog( parent )
{
	ui.setupUi(this);
	m_dCurrentTime = 0;
	ui.timeSlider->setMaximum( 100 );
	m_bAvoidSignal = false;
	connect( ui.timeSlider, SIGNAL( valueChanged( int ) ), this, SLOT( ChangeCurrentTime() ) );
	connect( ui.setTimeButton, SIGNAL( clicked() ), this, SLOT( SetSpecificTime() ) );
	connect(ui.timeEdit, SIGNAL(returnPressed()), this, SLOT(SetSpecificTime()));
	connect(ui.lastTimePoint, SIGNAL(clicked()), this, SLOT(GoToLastTimePoint()));
	connect(ui.firstTimePoint, SIGNAL(clicked()), this, SLOT(GoToFirstTimePoint()));
	connect(ui.nextTimePoint, SIGNAL(clicked()), this, SLOT(GoToNextTimePoint()));
	connect(ui.previousTimePoint, SIGNAL(clicked()), this, SLOT(GoToPreviousTimePoint()));
}

void CTimeSliderTab::ChangeCurrentTime()
{
	if (m_bAvoidSignal) return;
	SetTime(m_pSystemStructure->GetMaxTime() / ui.timeSlider->maximum()* ui.timeSlider->sliderPosition());
}

void CTimeSliderTab::SetSpecificTime()
{
	double dNewTime = GetConvValue(ui.timeEdit, EUnitType::TIME);
	if (dNewTime < 0)
		dNewTime = 0;
	else if (dNewTime > m_pSystemStructure->GetMaxTime())
		dNewTime = m_pSystemStructure->GetMaxTime();
	SetTime(dNewTime);
}

void CTimeSliderTab::GoToLastTimePoint()
{
	SetTime(m_pSystemStructure->GetMaxTime());
}

void CTimeSliderTab::GoToFirstTimePoint()
{
	SetTime(m_pSystemStructure->GetMinTime());
}

void CTimeSliderTab::GoToNextTimePoint()
{
	std::vector<double> vTimePoints = m_pSystemStructure->GetAllTimePoints();

	if (vTimePoints.empty())
		SetTime(0);
	else
		if (vTimePoints.back() <= m_dCurrentTime)
			SetTime(vTimePoints.back());
		else
			for (unsigned i = 0; i < vTimePoints.size(); i++)
				if (vTimePoints[i] > m_dCurrentTime)
				{
					SetTime(vTimePoints[i]);
					return;
				}
}

void CTimeSliderTab::GoToPreviousTimePoint()
{
	std::vector<double> vTimePoints = m_pSystemStructure->GetAllTimePoints();

	// dirty HACK to fix absence of first time point in saved vector
	double dMinTime = m_pSystemStructure->GetMinTime();
	if (!vTimePoints.empty() && vTimePoints.front() != dMinTime)
		vTimePoints.insert(vTimePoints.begin(), dMinTime);

	if (vTimePoints.empty() || (m_dCurrentTime == 0))
		SetTime(0);
	else
		for (unsigned i = 0; i < vTimePoints.size(); i++)
			if (vTimePoints[vTimePoints.size() - i - 1] < m_dCurrentTime)
			{
				SetTime(vTimePoints[vTimePoints.size() - i - 1]);
				return;
			}
}

void CTimeSliderTab::SetTime(double _dTime)
{
	m_bAvoidSignal = true;
	double dMaxTime = m_pSystemStructure->GetMaxTime();
	m_dCurrentTime = std::min(dMaxTime, _dTime);
	m_dCurrentTime = std::max(0., _dTime);

	ShowConvValue(ui.timeEdit, m_dCurrentTime, EUnitType::TIME);
	if (dMaxTime > 0)
		ui.timeSlider->setSliderPosition(m_dCurrentTime*ui.timeSlider->maximum() / dMaxTime);
	else
		ui.timeSlider->setSliderPosition(0);

	emit NewTimeSelected();
	m_bAvoidSignal = false;
}

void CTimeSliderTab::UpdateWholeView()
{
	ui.timeUnit->setText(ss2qs(m_pUnitConverter->GetSelectedUnit(EUnitType::TIME)));
	ChangeCurrentTime();
}

void CTimeSliderTab::SetTimeSliderEnabled()
{
	const auto tp = m_pSystemStructure->GetAllTimePoints();
	setEnabled(tp.size() >= 2 || m_pSystemStructure->GetAllTimePointsOldFormat().size() >= 2 || tp.size() == 1 && m_dCurrentTime > tp.front());
}


