/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_ObjectsGeneratorTab.h"
#include "AggloCompounds.h"

class CObjectsGeneratorTab : public CMusenDialog
{
	Q_OBJECT

	Ui::ObjectsGeneratorTab ui{};

	CGenerationManager* m_generationManager;
	CAggloCompounds m_aggloCompounds{ this };

public:
	CObjectsGeneratorTab(CGenerationManager* _generationManager, QWidget* _parent = nullptr);

	void UpdateWholeView() override;

private:
	void InitializeConnections() const;
	void UpdateGeneratorsList() const;

private slots:
	void AddGenerator();
	void DeleteGenerator();

	void GeneratorChanged() const;
	void GeneratorTypeChanged();
	void VelocityTypeChanged();

	void UpdateSelectedGenerator();
	void SetParameters();

	void SetupAggloCompounds();
	void NewAgglomerateChosen(int _index) const;
};
