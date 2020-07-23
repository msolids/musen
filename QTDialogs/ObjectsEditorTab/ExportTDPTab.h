/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_ExportTDPTab.h"
#include "SystemStructure.h"
#include "GeneralMUSENDialog.h"

class CExportTDPTab: public CMusenDialog
{
	Q_OBJECT
private:
	Ui::exportTDPTab ui;
	std::vector<size_t> m_vSelectedObjectsID;

	double m_dMinTime;
	double m_dMaxTime;
	double m_dTimeStep;
	QString m_sOutputFileName;

private:
	//void UpdateEntriesData();

public slots:
	void setVisible( bool _bVisible );
	void DataWasChanged();
	void ChangeFileName();
	void ExportData();

public:
	CExportTDPTab( QWidget *parent = 0 );
	void UpdateWholeView();
	void InitializeConnections();

	void SetSelectedObjectsID(const std::vector<size_t>& _vecObjectsID){ m_vSelectedObjectsID = _vecObjectsID; };
};
