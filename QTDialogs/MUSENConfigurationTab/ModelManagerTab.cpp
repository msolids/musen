/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelManagerTab.h"
#include "qtOperations.h"
#include <QFileDialog>

CModelManagerTab::CModelManagerTab(CModelManager* _pModelManager, QSettings* _pSettings, QWidget *parent)
	:CMusenDialog( parent ), m_pSettings( _pSettings )
{
	ui.setupUi(this);
	ui.modelsTable->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	ui.modelsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
	m_bAvoidSignal = false;
	m_pModelManager = _pModelManager;
	LoadConfiguration();
	InitializeConnections();
}

void CModelManagerTab::InitializeConnections()
{
	// signals from buttons
	connect(ui.addDir,		&QPushButton::clicked, this, &CModelManagerTab::AddDir);
	connect(ui.removeDir,	&QPushButton::clicked, this, &CModelManagerTab::RemoveDir);
	connect(ui.upDir,		&QPushButton::clicked, this, &CModelManagerTab::UpDir);
	connect(ui.downDir,		&QPushButton::clicked, this, &CModelManagerTab::DownDir);
	connect(ui.okButton,	&QPushButton::clicked, this, &CModelManagerTab::accept);
}

void CModelManagerTab::UpdateWholeView()
{
	UpdateFoldersView();
	UpdateModelsListView();
}

void CModelManagerTab::UpdateFoldersView()
{
	ui.directoryList->clear();
	std::vector<std::string> vFoldersList = m_pModelManager->GetDirs();
	for (int i = 0; i < (int)vFoldersList.size(); ++i)
		ui.directoryList->insertItem(i, ss2qs(vFoldersList[i]));
}

void CModelManagerTab::UpdateModelsListView() const
{
	ui.modelsTable->setSortingEnabled(false);
	ui.modelsTable->setRowCount(0);

	const auto vAllModels = m_pModelManager->GetAllAvailableModelsDescriptors();
	for (int i = 0; i < (int)vAllModels.size(); ++i)
	{
		ui.modelsTable->insertRow(ui.modelsTable->rowCount());

		// set name
		ui.modelsTable->SetItemNotEditable(i, 0, ss2qs(vAllModels[i]->GetModel()->GetName()));

		// set type
		switch (vAllModels[i]->GetModel()->GetType())
		{
		case EMusenModelType::PP:			ui.modelsTable->SetItemNotEditable(i, 1, tr("Particle-particle"));	break;
		case EMusenModelType::PW:			ui.modelsTable->SetItemNotEditable(i, 1, tr("Particle-wall"));		break;
		case EMusenModelType::SB:			ui.modelsTable->SetItemNotEditable(i, 1, tr("Solid bond"));			break;
		case EMusenModelType::LB:			ui.modelsTable->SetItemNotEditable(i, 1, tr("Liquid bond"));		break;
		case EMusenModelType::EF:			ui.modelsTable->SetItemNotEditable(i, 1, tr("External force"));		break;
		case EMusenModelType::PPHT:		    ui.modelsTable->SetItemNotEditable(i, 1, tr("PP heat transfer"));	break;
		case EMusenModelType::UNSPECIFIED:	ui.modelsTable->SetItemNotEditable(i, 1, tr("Unspecified"));		break;
		}

		// set path
		if (vAllModels[i]->GetLibType() == ELibType::STATIC)
			ui.modelsTable->SetItemNotEditable(i, 2, tr("Built-in"));
		else if (vAllModels[i]->GetLibType() == ELibType::DYNAMIC)
			ui.modelsTable->SetItemNotEditable(i, 2, ss2qs(vAllModels[i]->GetName()));
	}
	ui.modelsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
	ui.modelsTable->setSortingEnabled(true);
}

void CModelManagerTab::AddDir()
{
	if (m_bAvoidSignal) return;
	QString sFolderPath = QFileDialog::getExistingDirectory(this, "Open Directory", ".", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
	if (!sFolderPath.isEmpty())
	{
		m_pModelManager->AddDir(qs2ss(sFolderPath));
		UpdateWholeView();
	}
}

void CModelManagerTab::RemoveDir()
{
	int iRow = ui.directoryList->currentRow();
	if (iRow < 0) return;
	m_pModelManager->RemoveDir(ui.directoryList->currentRow());
	UpdateWholeView();
}

void CModelManagerTab::UpDir()
{
	int iRow = ui.directoryList->currentRow();
	if (iRow < 0) return;
	m_pModelManager->UpDir(iRow);
	UpdateWholeView();
	ui.directoryList->setCurrentRow(iRow == 0 ? 0 : iRow - 1);
}

void CModelManagerTab::DownDir()
{
	int iRow = ui.directoryList->currentRow();
	if (iRow < 0) return;
	int iLastRow = ui.directoryList->count() - 1;
	m_pModelManager->DownDir(iRow);
	UpdateWholeView();
	ui.directoryList->setCurrentRow(iRow == iLastRow ? iLastRow : iRow + 1);
}

void CModelManagerTab::LoadConfiguration()
{
	std::vector<std::string> vFoldersList;
	int nSize = m_pSettings->beginReadArray(MM_DLL_FOLDER_NAME);
	for (int i = 0; i < nSize; ++i)
	{
		m_pSettings->setArrayIndex(i);
		vFoldersList.push_back(m_pSettings->value(MM_DLL_FOLDER_NAME).toString().toStdString());
	}
	m_pSettings->endArray();

	m_pModelManager->SetDirs(vFoldersList);
}

void CModelManagerTab::SaveConfiguration()
{
	std::vector<std::string> vFoldesList = m_pModelManager->GetDirs();
	m_pSettings->beginWriteArray(MM_DLL_FOLDER_NAME);
	for (int i = 0; i < vFoldesList.size(); ++i)
	{
		m_pSettings->setArrayIndex(i);
		m_pSettings->setValue(MM_DLL_FOLDER_NAME, QString::fromStdString(vFoldesList[i]));
	}
	m_pSettings->endArray();
}
