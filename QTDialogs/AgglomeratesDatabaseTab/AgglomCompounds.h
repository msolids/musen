/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_AgglomCompounds.h"
#include "qtOperations.h"
#include "AgglomeratesDatabase.h"
#include "MaterialsDatabase.h"

class CAgglomCompounds : public QDialog
{
	Q_OBJECT

private:
	Ui::CAgglomCompounds ui;

	SAgglomerate* m_pAgglomerate;
	CMaterialsDatabase* m_pMaterialsDB;
	QList<QString> m_vPartKeys;
	QList<QString> m_vBondKeys;

public:
	CAgglomCompounds(SAgglomerate* _pAgglomerate, CMaterialsDatabase* _pMaterialsDB, QList<QString> _vPartKeys, QList<QString> _vBondKeys, QWidget *parent = 0);
	~CAgglomCompounds();

	void UpdateWholeView();

public slots:
	void setVisible(bool _bVisible);

private slots:
	void OkButtonClicked();
};
