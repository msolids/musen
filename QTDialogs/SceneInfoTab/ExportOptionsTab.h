/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_ExportOptionsTab.h"
#include "qtOperations.h"

class CExportOptionsTab: public QDialog
{
	Q_OBJECT
private:
	Ui::exportOptionsTab ui;
public:
	double m_dStartTime;
	double m_dEndTime;
	double m_dTimeStep;
	QString	m_sSeparator;
	unsigned m_nNumberOfPSDClasses;

private:
	void UpdateEntriesData();

public slots:
	void setVisible( bool _bVisible );
	void DataWasChanged();

public:
	CExportOptionsTab( QWidget *parent = 0 );
	void UpdateWholeView();
	void InitializeConnections();

};
