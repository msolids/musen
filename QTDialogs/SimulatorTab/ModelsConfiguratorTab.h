/* Copyright (c) 2013-2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_ModelsConfiguratorTab.h"
#include "ModelManager.h"

class CModelWidget;

class CModelsConfiguratorTab : public QDialog
{
	Q_OBJECT

	Ui::CModelsConfiguratorTab ui{};

	CModelManager* m_modelManager; // Pointer to a models manager.
	bool m_blockModelsChange;      // If set, modification of the models will be blocked.

	std::map<EMusenModelType, QBoxLayout*> m_layouts; // List of layouts where model widgets of different types are shown.

public:
	CModelsConfiguratorTab(CModelManager* _modelManager, bool _blockModelsChange, QWidget* _parent = nullptr);

	// Changes visibility of the widget.
	void setVisible(bool _visible) override;
	// Updates all child widgets.
	void UpdateWholeView();

	// Shows the dialog as a modal dialog.
	int exec() override;

private:
	// Connects signals and slots.
	void InitializeConnections();

	// Updates the view of currently selected models.
	void UpdateSelectedModels();

	// Called when a new model added.
	void AddModelClicked(EMusenModelType _type);
	// Called when a model removal requested.
	void RemoveModelClicked(EMusenModelType _type, CModelWidget* _widget);

	// Adds a new model widget of the given type. _modelDescriptor can be nullptr, if no model selected yet.
	void AddModelWedget(EMusenModelType _type, CModelDescriptor* _modelDescriptor);
};