/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_ModelWidget.h"
#include "AbstractDEMModel.h"

class CModelManager;
class CModelDescriptor;

class CModelWidget : public QWidget
{
	Q_OBJECT

	Ui::CModelWidgetClass ui{};

	CModelManager* m_modelManager;                               // Pointer to a models manager. Can be nullptr if no model selected yet.
	EMusenModelType m_modelType{ EMusenModelType::UNSPECIFIED }; // Type of the models this widget works with.
	CModelDescriptor* m_modelDescriptor{};                       // Pointer to a model descriptor that is shown in the widget.

public:
	CModelWidget(CModelManager* _modelManager, EMusenModelType _type, CModelDescriptor* _modelDescriptor, QWidget* _parent = nullptr);

	// Returns a pointer to a descriptor of a currently selected model.
	CModelDescriptor* GetModelDescriptor() const;

	// Changes visibility of the widget.
	void setVisible(bool _visible) override;
	// Updates all child widgets.
	void UpdateWholeView() const;

private:
	// Connects signals and slots.
	void InitializeConnections();

	// Fills the models combobox with required entries.
	void CreateModelsCombo() const;

	// Updates the models combobox with the currently selected model.
	void UpdateModelsCombo() const;
	// Updates activity of the config button according to the currently selected model.
	void UpdateConfigButton() const;

	// Called when a new model is selected.
	void ModelChanged();
	// Called when the config button is clicked.
	void ConfigButtonClicked() const;
	// Called when the help button is clicked.
	void HelpButtonClicked() const;

	// Returns true if current model has parameters.
	bool IsModelHasParams() const;

signals:
	// Emitted when remove button is clicked.
	void RemoveRequested();
};
