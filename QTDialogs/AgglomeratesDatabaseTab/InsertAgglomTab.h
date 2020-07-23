/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "qtOperations.h"
#include "ui_InsertAgglomTab.h"
#include "SystemStructure.h"
#include "AgglomeratesDatabase.h"
#include "AgglomOpenGLView.h"
#include "UnitConvertor.h"
#include "GeneralMUSENDialog.h"

class CInsertAgglomTab : public CMusenDialog
{
	Q_OBJECT

private:
	Ui::insertAgglomTab ui;
	std::string m_sAgglomKey;

public:
	CInsertAgglomTab(QWidget *parent = 0);

	void SetCurrentAgglom(const std::string& _sKey);

private:
	QComboBox* CreateMaterialCombo(QWidget* _pParent);

private slots:
	void UpdateWholeView();
	void AddAgglomerate();

signals:
	void NewAgglomerateAdded();
};