/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelsConfiguratorTab.h"
#include <QDesktopServices>

CModelsConfiguratorTab::CModelsConfiguratorTab(CModelManager* _pModelManager, bool _bBlockModelsChange, QWidget *parent /*= 0*/) : QDialog(parent)
{
	ui.setupUi(this);
	ui.label_6->setVisible(false);
	ui.lbDll->setVisible(false);
	ui.specModelParLB->setVisible(false);
	ui.helpLBModel->setVisible(false);
	m_pModelManager = _pModelManager;
	m_bBlockModelsChange = _bBlockModelsChange;
	m_vCombos[EMusenModelType::PP] = ui.ppDll;
	m_vCombos[EMusenModelType::PW] = ui.pwDll;
	m_vCombos[EMusenModelType::SB] = ui.sbDll;
	m_vCombos[EMusenModelType::LB] = ui.lbDll;
	m_vCombos[EMusenModelType::EF] = ui.efDll;
	m_vCombos[EMusenModelType::PPHT] = ui.htPPdll;
	InitializeConnections();
}

CModelsConfiguratorTab::~CModelsConfiguratorTab()
{
}

void CModelsConfiguratorTab::InitializeConnections()
{
	connect(ui.okButton, &QPushButton::clicked, this, &QDialog::accept);

	// signals from buttons to specify model parameters
	connect(ui.specModelParPP, &QPushButton::clicked, this, [this] {SpecModelParameters(EMusenModelType::PP); });
	connect(ui.specModelParPW, &QPushButton::clicked, this, [this] {SpecModelParameters(EMusenModelType::PW); });
	connect(ui.specModelParSB, &QPushButton::clicked, this, [this] {SpecModelParameters(EMusenModelType::SB); });
	connect(ui.specModelParLB, &QPushButton::clicked, this, [this] {SpecModelParameters(EMusenModelType::LB); });
	connect(ui.specModelParEF, &QPushButton::clicked, this, [this] {SpecModelParameters(EMusenModelType::EF); });
	connect(ui.specModelParHT_PP, &QPushButton::clicked, this, [this] {SpecModelParameters(EMusenModelType::PPHT); });

	// signals from buttons to see help
	connect(ui.helpPPModel, &QPushButton::clicked, this, [this] {OpenModelDocumentation(EMusenModelType::PP); });
	connect(ui.helpPWModel, &QPushButton::clicked, this, [this] {OpenModelDocumentation(EMusenModelType::PW); });
	connect(ui.helpSBModel, &QPushButton::clicked, this, [this] {OpenModelDocumentation(EMusenModelType::SB); });
	connect(ui.helpLBModel, &QPushButton::clicked, this, [this] {OpenModelDocumentation(EMusenModelType::LB); });
	connect(ui.helpEFModel, &QPushButton::clicked, this, [this] {OpenModelDocumentation(EMusenModelType::EF); });
	connect(ui.helpHT_PPModel, &QPushButton::clicked, this, [this] {OpenModelDocumentation(EMusenModelType::PPHT); });

	// signals from model combo boxes
	connect(ui.ppDll, SIGNAL(currentIndexChanged(int)), this, SLOT(SelectedModelsChanged()));
	connect(ui.pwDll, SIGNAL(currentIndexChanged(int)), this, SLOT(SelectedModelsChanged()));
	connect(ui.sbDll, SIGNAL(currentIndexChanged(int)), this, SLOT(SelectedModelsChanged()));
	connect(ui.lbDll, SIGNAL(currentIndexChanged(int)), this, SLOT(SelectedModelsChanged()));
	connect(ui.efDll, SIGNAL(currentIndexChanged(int)), this, SLOT(SelectedModelsChanged()));
	connect(ui.htPPdll, SIGNAL(currentIndexChanged(int)), this, SLOT(SelectedModelsChanged()));
}

void CModelsConfiguratorTab::setVisible(bool _bVisible)
{
	if (_bVisible)
		UpdateWholeView();
	QDialog::setVisible(_bVisible);
}

void CModelsConfiguratorTab::UpdateWholeView()
{
	UpdateSelectedModelsView();
}

int CModelsConfiguratorTab::exec()
{
	activateWindow();
	QDialog::exec();
	return 1;
}

void CModelsConfiguratorTab::UpdateSelectedModelsView()
{
	m_bAvoidSignal = true;

	// clear all combos and add possibility to exclude model ("")
	for (const auto& combo : m_vCombos)
	{
		combo.second->clear();
		combo.second->addItem("");
	}

	// put all models into combos
	const auto& allModels = m_pModelManager->GetAllAvailableModelsDescriptors();
	for (const auto& model : allModels)
	{
		std::string displayName;
		switch (model->GetLibType())
		{
		case ELibType::STATIC:	displayName = model->GetModel()->GetName();									break;
		case ELibType::DYNAMIC:	displayName = model->GetModel()->GetName() + " (" + model->GetName() + ")";	break;
		}
		m_vCombos[model->GetModel()->GetType()]->addItem(QString::fromStdString(displayName), QString::fromStdString(model->GetName()));
	}
	// set selected model
	for (auto& combo : m_vCombos)
	{
		const auto& models = m_pModelManager->GetModelsDescriptors(combo.first);
		combo.second->setCurrentIndex(combo.second->findData(QString::fromStdString(!models.empty() ? models.front()->GetName() : "")));
	}

	if (m_bBlockModelsChange)
		for (const auto& combo : m_vCombos)
			combo.second->setEnabled(false);

	UpdateConfigButtons();

	m_bAvoidSignal = false;
}

void CModelsConfiguratorTab::UpdateConfigButtons() const
{
	const auto SetParamButtonActive = [&](QPushButton* _button, EMusenModelType _type)
	{
		const auto& descriptors = m_pModelManager->GetModelsDescriptors(_type);
		_button->setEnabled(!descriptors.empty() && descriptors.front()->GetModel() && descriptors.front()->GetModel()->GetParametersNumber() != 0);
	};

	SetParamButtonActive(ui.specModelParPP   , EMusenModelType::PP);
	SetParamButtonActive(ui.specModelParPW   , EMusenModelType::PW);
	SetParamButtonActive(ui.specModelParSB   , EMusenModelType::SB);
	SetParamButtonActive(ui.specModelParLB   , EMusenModelType::LB);
	SetParamButtonActive(ui.specModelParEF   , EMusenModelType::EF);
	SetParamButtonActive(ui.specModelParHT_PP, EMusenModelType::PPHT);
}

void CModelsConfiguratorTab::SelectedModelsChanged() const
{
	if (m_bAvoidSignal) return;

	for (auto& combo : m_vCombos)
	{
		const auto& descriptors = m_pModelManager->GetModelsDescriptors(combo.first);
		if (!descriptors.empty())
		{
			m_pModelManager->RemoveActiveModel(descriptors.front()->GetName());
			//const std::string oldModelName = combo.second->currentData().toString().toStdString();
			//if (descriptors.front()->GetName() != oldModelName)	                                                         // not the same model
			//	m_pModelManager->ReplaceActiveModel(oldModelName, combo.second->currentData().toString().toStdString()); // set new model
		}
		m_pModelManager->AddActiveModel(combo.second->currentData().toString().toStdString());
	}
	UpdateConfigButtons();
}

void CModelsConfiguratorTab::SpecModelParameters(const EMusenModelType& _modelType) const
{
	if (!m_pModelManager->IsModelActive(_modelType)) return; // no model specified

	const auto descriptors = m_pModelManager->GetModelsDescriptors(_modelType);
	CModelParameterTab paramEditor(descriptors.front()->GetModel());
	paramEditor.exec();
}

void CModelsConfiguratorTab::OpenModelDocumentation(const EMusenModelType& _modelType) const
{
	if (!m_pModelManager->IsModelActive(_modelType))	return; // no model specified

	const auto& descriptors = m_pModelManager->GetModelsDescriptors(_modelType);
	const QString helpFileName = QString::fromStdString(descriptors.front()->GetModel()->GetHelpFileName());
	if (helpFileName != "")
		QDesktopServices::openUrl(QUrl::fromLocalFile("file:///" + QCoreApplication::applicationDirPath() + "/Documentation/Models" + helpFileName));
}
