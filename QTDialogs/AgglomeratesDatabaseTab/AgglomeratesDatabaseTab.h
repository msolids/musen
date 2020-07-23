/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_AgglomeratesDatabaseTab.h"
#include "AgglomOpenGLView.h"
#include "InsertAgglomTab.h"
#include "AgglomCompounds.h"

class CAgglomeratesDatabaseTab : public CMusenDialog
{
	Q_OBJECT
private:
	Ui::agglomeratesDatabaseTab ui;
	CInsertAgglomTab* m_pInsertAgglomTab;
	bool m_bEnableInsertion;

public:
	CAgglomeratesDatabaseTab(QWidget *parent = 0);

	void SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB);
	void EnableInsertion(bool _bEnable);

private:
	void InitializeConnections();
	void UpdateButtons();
	void UpdateSelectedAgglomInfo();

private slots:
	void NewDatabase();
	void SaveDatabase();
	void SaveDatabaseAs( const QString& _sFileName = "" );
	void LoadDatabase();
	void DataWasChanged();
	void ShowAgglomerate();
	void DeleteAgglomerate();
	void UpAgglomerate();
	void DownAgglomerate();

	void AddAgglomerate();
	void NewAgglomerateAdded();
	void UpdateWholeView();
	void NewRowSelected();
	void InsertAgglomerate();

signals:
	void AgglomerateAdded();
};