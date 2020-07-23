/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzerTab.h"
#include "AgglomeratesAnalyzer.h"

class CAgglomeratesAnalyzerTab : public CResultsAnalyzerTab
{
	Q_OBJECT

public:
	CAgglomeratesAnalyzerTab(QWidget *parent);
	void InitializeAnalyzerTab() override;

public slots:
	void UpdateWholeView() override;
};
