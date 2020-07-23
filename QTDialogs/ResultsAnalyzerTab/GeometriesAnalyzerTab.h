/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ResultsAnalyzerTab.h"
#include "GeometriesAnalyzer.h"

class CGeometriesAnalyzerTab : public CResultsAnalyzerTab
{
	Q_OBJECT
public:
	CGeometriesAnalyzerTab(QWidget *parent);
	void InitializeAnalyzerTab();

private:
	void UpdateDistanceVisibility() override;


public slots:
	void UpdateWholeView() override;
};