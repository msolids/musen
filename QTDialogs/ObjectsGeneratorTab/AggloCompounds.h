/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_AggloCompounds.h"
#include "SystemStructure.h"
#include "GenerationManager.h"
#include "qtOperations.h"
#include <QtWidgets/QComboBox>

class CAggloCompounds : public QDialog
{
	Q_OBJECT

private:
	Ui::CAggloCompounds ui;

	CSystemStructure* m_pSystemStructure;
	CObjectsGenerator* m_pGenerator;
	CAgglomeratesDatabase* m_pAgglomeratesDB;

public:
	CAggloCompounds(QWidget *parent = 0);
	~CAggloCompounds();

	void SetPointers(CSystemStructure* _pSystemStructure, CAgglomeratesDatabase* _pAgglomeratesDB);
	void UpdateWholeView();
	void SetGenerator(CObjectsGenerator* _pGenerator);

public slots:
	void setVisible(bool _bVisible);

private:
	QComboBox* CreateMaterialCombo(QWidget* _pParent, const QString& _sAlias);

private slots:
	void OkButtonClicked();
};
