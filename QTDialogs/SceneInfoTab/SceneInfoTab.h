/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_SceneInfoTab.h"
#include "AgglomeratesAnalyzer.h"
#include "SystemStructure.h"
#include "GeneralMUSENDialog.h"
#include "ExportOptionsTab.h"
#include "UnitConvertor.h"

class CSceneInfoTab: public CMusenDialog
{
	Q_OBJECT
private:
	Ui::sceneInfoTab ui;
	CExportOptionsTab m_ExportOptionsTab;

private:
	void UpdateMaxOverlap(); // updates maximal overlap between particles
	void closeEvent( QCloseEvent * event );
	void InitializeConnections();
	void UpdateLabelsHeaders();
private slots:
	void UpdateInfo();
	void UpdateDetailedInfo();

public slots:
	void UpdateComboBoxList(); // update the list of existing materials in the combo box
	void UpdateTableView(); // update information in the table view
	void UpdateAgglomNumber();
	void ExportResults();

public:
	CSceneInfoTab( QWidget *parent = 0 );
};


