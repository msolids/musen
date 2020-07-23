/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "qtOperations.h"
#include "AbstractDEMModel.h"
#include "ui_ModelParameterTab.h"

class CModelParameterTab: public QDialog
{
	Q_OBJECT
private:
	Ui::modelParameterTab ui;
	CAbstractDEMModel* m_pModel;

public:
	CModelParameterTab(CAbstractDEMModel* _pModel, QWidget* _pParent = NULL);

public slots:
	void setVisible(bool _bVisible);

private:
	void InitializeConnections();
	void CreateInitialForms();		// Creates all necessary forms for parameters.

	void UpdateWholeView();
	void UpdateParametersView();	// Updates values of all parameters.

private slots:
	void SetDefaultParameters();
	void SetNewParameters();
};
