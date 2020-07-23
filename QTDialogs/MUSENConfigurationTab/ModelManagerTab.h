/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ModelManager.h"
#include "GeneralMUSENDialog.h"
#include "ui_ModelManagerTab.h"
#include <QSettings>

#define MM_DLL_FOLDER_NAME "MM_DLL_FOLDER_NAME"

class CModelManagerTab : public CMusenDialog
{
	Q_OBJECT
private:
	Ui::modelManagerTab ui;

	CModelManager* m_pModelManager;
	QSettings* m_pSettings;

public:
	CModelManagerTab(CModelManager* _pModelManager, QSettings* _pSettings, QWidget *parent = 0);

	void SaveConfiguration();
	void LoadConfiguration();

public slots:
	void UpdateWholeView();

private:
	void InitializeConnections();
	void UpdateFoldersView();
	void UpdateModelsListView();

private slots:
	void AddDir();
	void RemoveDir();
	void UpDir();
	void DownDir();
};
