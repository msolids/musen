/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_UnitConverterTab.h"
#include <QComboBox>
#include <QSettings>

class CUnitConvertorTab : public CMusenDialog
{
	Q_OBJECT
private:
	std::vector<QString> m_vFoldersList;
	std::vector<QComboBox*> m_pComboBoxes;
	CUnitConvertor* m_pUnitConvertor;
	Ui::unitConverterTab ui;
	QSettings* m_pSettings;

private:
	void InitializeConnections();

public slots:
	void UpdateWholeView();
	void SetNewUnits();
	void SetNewUnitsAndClose();
	void RestoreDefaultUnits();

public:
	CUnitConvertorTab( QSettings* _pSettings, QWidget *parent = 0 );

	void SetPointers( CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB );

	void SaveConfiguration();
	void LoadConfiguration();

signals:
	void NewUnitsSelected();
};
