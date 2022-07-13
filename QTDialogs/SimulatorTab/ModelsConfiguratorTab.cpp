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
	for (auto it = m_vCombos.begin(); it != m_vCombos.end(); ++it)
	{
		it->second->clear();
		it->second->addItem("");
	}

	// put all models into combos
	std::vector<CModelManager::SModelInfo> vAllModels = m_pModelManager->GetAllAvailableModels();
	for (size_t i = 0; i < vAllModels.size(); ++i)
	{
		if (vAllModels[i].libType == CModelManager::ELibType::STATIC)
			m_vCombos[vAllModels[i].pModel->GetType()]->addItem(ss2qs(vAllModels[i].pModel->GetName()), ss2qs(vAllModels[i].sPath));
		else if (vAllModels[i].libType == CModelManager::ELibType::DYNAMIC)
			m_vCombos[vAllModels[i].pModel->GetType()]->addItem(ss2qs(vAllModels[i].pModel->GetName() + " (" + vAllModels[i].sPath + ")"), ss2qs(vAllModels[i].sPath));
	}
	// set selected model
	for (auto it = m_vCombos.begin(); it != m_vCombos.end(); ++it)
		it->second->setCurrentIndex(it->second->findData(ss2qs(m_pModelManager->GetModelPath(it->first))));

	if(m_bBlockModelsChange)
		for (auto it = m_vCombos.begin(); it != m_vCombos.end(); ++it)
			it->second->setEnabled(false);

	UpdateConfigButtons();

	m_bAvoidSignal = false;
}

void CModelsConfiguratorTab::UpdateConfigButtons()
{
	CAbstractDEMModel *pPP = m_pModelManager->GetModel(EMusenModelType::PP);
	CAbstractDEMModel *pPW = m_pModelManager->GetModel(EMusenModelType::PW);
	CAbstractDEMModel *pSB = m_pModelManager->GetModel(EMusenModelType::SB);
	CAbstractDEMModel *pLB = m_pModelManager->GetModel(EMusenModelType::LB);
	CAbstractDEMModel *pEF = m_pModelManager->GetModel(EMusenModelType::EF);
	CAbstractDEMModel *pHTPP = m_pModelManager->GetModel(EMusenModelType::PPHT);
	ui.specModelParPP->setEnabled(pPP && pPP->GetParametersNumber() != 0);
	ui.specModelParPW->setEnabled(pPW && pPW->GetParametersNumber() != 0);
	ui.specModelParSB->setEnabled(pSB && pSB->GetParametersNumber() != 0);
	ui.specModelParLB->setEnabled(pLB && pLB->GetParametersNumber() != 0);
	ui.specModelParEF->setEnabled(pEF && pEF->GetParametersNumber() != 0);
	ui.specModelParHT_PP->setEnabled(pHTPP && pHTPP->GetParametersNumber() != 0);
}

void CModelsConfiguratorTab::SelectedModelsChanged()
{
	if (m_bAvoidSignal) return;

	for (auto it = m_vCombos.begin(); it != m_vCombos.end(); ++it)
		if (ss2qs(m_pModelManager->GetModelPath(it->first)) != it->second->currentData().toString())	// not the same model
			m_pModelManager->SetModelPath(it->first, qs2ss(it->second->currentData().toString()));		// set new model
	UpdateConfigButtons();
}

void CModelsConfiguratorTab::SpecModelParameters(const EMusenModelType& _modelType)
{
	if (!m_pModelManager->IsModelDefined(_modelType)) return; // no model specified

	CModelParameterTab paramEditor(m_pModelManager->GetModel(_modelType));
	paramEditor.exec();
}

void CModelsConfiguratorTab::OpenModelDocumentation(const EMusenModelType& _modelType)
{
	if (!m_pModelManager->IsModelDefined(_modelType))	return; // no model specified

	QString sHelpFileName = ss2qs(m_pModelManager->GetModel(_modelType)->GetHelpFileName());
	if (sHelpFileName != "")
		QDesktopServices::openUrl(QUrl::fromLocalFile("file:///" + QCoreApplication::applicationDirPath() + "/Documentation/Models" + sHelpFileName));
}
