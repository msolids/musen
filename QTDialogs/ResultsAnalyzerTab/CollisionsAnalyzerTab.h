/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzerTab.h"

class CCollisionsAnalyzerTab : public CResultsAnalyzerTab
{
	Q_OBJECT
private:
	QSize m_size;
public:
	CCollisionsAnalyzerTab(QWidget *parent);
	void Initialize();

public slots:
	void UpdateWholeView() override;
};
