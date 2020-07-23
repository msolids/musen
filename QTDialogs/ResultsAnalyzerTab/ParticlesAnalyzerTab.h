/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzerTab.h"
#include "ParticlesAnalyzer.h"

class CParticlesAnalyzerTab : public CResultsAnalyzerTab
{
	Q_OBJECT
public:
	CParticlesAnalyzerTab(QWidget *parent);
	void InitializeAnalyzerTab();
public slots:
	void UpdateWholeView() override;
};
