/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_ObjectsGeneratorTab.h"
#include "AgglomeratesDatabase.h"
#include "AggloCompounds.h"

class CObjectsGeneratorTab : public CMusenDialog
{
	Q_OBJECT

public:
	CObjectsGeneratorTab(CGenerationManager* _pGenerationManager, QWidget *parent = 0);

private:
	Ui::objectsGeneratorTab ui;

	CGenerationManager* m_pGenerationManager;
	CAggloCompounds* m_pAggloCompounds;
	std::vector<double> m_vMagnitudes;

private:
	void InitializeConnections() const;
	void UpdateGeneratorsList();

private slots:
	void GeneratorWasChanged();
	void UpdateSelectedGenerator();
	void SetParameters();
	void GeneratorTypeChanged();
	void VelocityTypeChanged();

	void DeleteGenerator();
	void AddGenerator();

	void UpdateWholeView();
	void SetupAggloCompounds();
	void NewAgglomerateChosen(int _nIndex);
};
