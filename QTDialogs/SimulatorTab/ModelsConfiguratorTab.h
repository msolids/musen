/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ModelManager.h"
#include "ModelParameterTab.h"
#include "ui_ModelsConfiguratorTab.h"

class CModelsConfiguratorTab : public QDialog
{
	Q_OBJECT
private:
	Ui::CModelsConfiguratorTab ui;

	CModelManager* m_pModelManager;
	bool m_bBlockModelsChange;

	std::map<EMusenModelType, QComboBox*> m_vCombos;	// List of comboboxes for all selected models.

	bool m_bAvoidSignal;

public:
	CModelsConfiguratorTab(CModelManager* _pModelManager, bool _bBlockModelsChange, QWidget *parent = 0);
	~CModelsConfiguratorTab();

public slots:
	void setVisible(bool _bVisible);
	void UpdateWholeView();
	int exec();

private:
	void InitializeConnections();
	void UpdateSelectedModelsView();
	void UpdateConfigButtons() const;

private slots:
	void SelectedModelsChanged() const;
	void SpecModelParameters(const EMusenModelType& _modelType) const;
	void OpenModelDocumentation(const EMusenModelType& _modelType) const;
};
