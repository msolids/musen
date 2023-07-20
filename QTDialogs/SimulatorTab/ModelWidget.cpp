/* Copyright (c) 2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelWidget.h"
#include "ModelManager.h"
#include "ModelParameterTab.h"
#include "QtSignalBlocker.h"
#include <QDesktopServices>

CModelWidget::CModelWidget(CModelManager* _modelManager, EMusenModelType _type, CModelDescriptor* _modelDescriptor, QWidget* _parent)
	: QWidget{ _parent }
	, m_modelManager{ _modelManager }
	, m_modelType{ _type }
	, m_modelDescriptor{ _modelDescriptor }
{
	ui.setupUi(this);
	CreateModelsCombo();
	InitializeConnections();
}

CModelDescriptor* CModelWidget::GetModelDescriptor() const
{
	return m_modelDescriptor;
}

void CModelWidget::InitializeConnections()
{
	connect(ui.comboModel  , QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CModelWidget::ModelChanged);
	connect(ui.buttonConfig, &QPushButton::clicked                              , this, &CModelWidget::ConfigButtonClicked);
	connect(ui.buttonHelp  , &QPushButton::clicked                              , this, &CModelWidget::HelpButtonClicked);
	connect(ui.buttonRemove, &QPushButton::clicked                              , this, &CModelWidget::RemoveRequested);
}

void CModelWidget::setVisible(bool _visible)
{
	if (_visible)
		UpdateWholeView();
	QWidget::setVisible(_visible);
}

void CModelWidget::CreateModelsCombo() const
{
	[[maybe_unused]] CQtSignalBlocker blocker{ ui.comboModel };

	for (const auto& modelDescriptor : m_modelManager->GetAvailableModelsDescriptors(m_modelType))
	{
		std::string displayName;
		switch (modelDescriptor->GetLibType())
		{
		case ELibType::STATIC:	displayName = modelDescriptor->GetModel()->GetName();                                           break;
		case ELibType::DYNAMIC:	displayName = modelDescriptor->GetModel()->GetName() + " (" + modelDescriptor->GetPath() + ")";	break;
		}
		ui.comboModel->addItem(QString::fromStdString(displayName), QString::fromStdString(modelDescriptor->GetPath()));
	}

	UpdateModelsCombo();
}

void CModelWidget::UpdateModelsCombo() const
{
	[[maybe_unused]] CQtSignalBlocker blocker{ ui.comboModel };

	if (!m_modelDescriptor || !m_modelDescriptor->GetModel())
	{
		ui.comboModel->setCurrentIndex(-1);
		return;
	}

	ui.comboModel->setCurrentIndex(ui.comboModel->findData(QString::fromStdString(m_modelDescriptor->GetPath())));
}

void CModelWidget::UpdateConfigButton() const
{
	[[maybe_unused]] CQtSignalBlocker blocker{ ui.buttonConfig };

	ui.buttonConfig->setEnabled(IsModelHasParams());
}

void CModelWidget::UpdateWholeView() const
{
	UpdateModelsCombo();
	UpdateConfigButton();
}

void CModelWidget::ModelChanged()
{
	const std::string oldModelName = m_modelDescriptor ? m_modelDescriptor->GetPath() : "";
	const std::string newModelName = ui.comboModel->currentData().toString().toStdString();
	if (newModelName == oldModelName) return;

	m_modelDescriptor = m_modelManager->ReplaceActiveModel(oldModelName, newModelName);

	UpdateConfigButton();
}

void CModelWidget::ConfigButtonClicked() const
{
	if (!IsModelHasParams()) return;

	CModelParameterTab paramEditor(m_modelDescriptor->GetModel());
	paramEditor.exec();
}

void CModelWidget::HelpButtonClicked() const
{
	if (!m_modelDescriptor || m_modelDescriptor->GetModel()) return;

	const QString helpFile = QString::fromStdString(m_modelDescriptor->GetModel()->GetHelpFileName());
	if (!helpFile.isEmpty())
		QDesktopServices::openUrl(QUrl::fromLocalFile("file:///" + QCoreApplication::applicationDirPath() + "/Documentation/Models" + helpFile));
}

bool CModelWidget::IsModelHasParams() const
{
	return m_modelDescriptor && m_modelDescriptor->GetModel() && m_modelDescriptor->GetModel()->GetParametersNumber() != 0;
}

