/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BondsGeneratorTab.h"
#include "QtSignalBlocker.h"
#include <QMessageBox>

CBondsGeneratorProcess::CBondsGeneratorProcess(CBondsGenerator* _bondsGenerator, QObject* _parent /*= 0*/) :
	QObject{ _parent },
	m_bondsGenerator{ _bondsGenerator }
{}

void CBondsGeneratorProcess::StartGeneration()
{
	m_bondsGenerator->StartGeneration();
	emit finished();
}

void CBondsGeneratorProcess::StopGeneration() const
{
	if (m_bondsGenerator->Status() == ERunningStatus::RUNNING)
		m_bondsGenerator->SetStatus(ERunningStatus::TO_BE_STOPPED);
}


CBondsGeneratorTab::CBondsGeneratorTab(CBondsGenerator* _bondsGenerator, QWidget* _parent) :
	CMusenDialog(_parent),
	m_generator{ _bondsGenerator },
	m_generatorProcess{ m_generator }
{
	ui.setupUi(this);
	ui.tableClasses->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	ui.tableClasses->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);

	m_sHelpFileName = "Users Guide/Bonds Generator.pdf";

	m_generatorProcess.moveToThread(&m_generatorThread);

	InitializeConnections();
}

void CBondsGeneratorTab::InitializeConnections() const
{
	// classes manipulation buttons
	connect(ui.buttonAddClass,    &QPushButton::clicked, this, &CBondsGeneratorTab::AddClass);
	connect(ui.buttonRemoveClass, &QPushButton::clicked, this, &CBondsGeneratorTab::RemoveClass);
	connect(ui.buttonUpClass,	  &QPushButton::clicked, this, &CBondsGeneratorTab::UpClass);
	connect(ui.buttonDownClass,   &QPushButton::clicked, this, &CBondsGeneratorTab::DownClass);

	// classes table
	connect(ui.tableClasses, &CQtTable::itemChanged,          this, &CBondsGeneratorTab::BondClassChanged);
	connect(ui.tableClasses, &CQtTable::CheckBoxStateChanged, this, &CBondsGeneratorTab::BondClassChanged);
	connect(ui.tableClasses, &CQtTable::ComboBoxIndexChanged, this, &CBondsGeneratorTab::BondClassChanged);
	connect(ui.tableClasses, &CQtTable::currentCellChanged,   this, &CBondsGeneratorTab::BondClassSelected);
	connect(ui.tableClasses, &CQtTable::EmptyAreaClicked,     this, &CBondsGeneratorTab::BondClassDeselected);

	// materials
	connect(ui.listMaterials1, &QListWidget::itemChanged, this, &CBondsGeneratorTab::SelectedMaterialsChanged);
	connect(ui.listMaterials2, &QListWidget::itemChanged, this, &CBondsGeneratorTab::SelectedMaterialsChanged);

	// buttons
	connect(ui.buttonDeleteBonds,	  &QPushButton::clicked, this, &CBondsGeneratorTab::DeleteBondsClicked);
	connect(ui.buttonStartGeneration, &QPushButton::clicked, this, &CBondsGeneratorTab::StartStopClicked);

	// threads and timers
	connect(&m_generatorThread,  &QThread::started,                 &m_generatorProcess, &CBondsGeneratorProcess::StartGeneration);
	connect(&m_generatorProcess, &CBondsGeneratorProcess::finished, this,                &CBondsGeneratorTab::GenerationFinished);
	connect(&m_statisticsTimer,  &QTimer::timeout,                  this,                &CBondsGeneratorTab::UpdateGenerationStatistics);
}

void CBondsGeneratorTab::UpdateWholeView()
{
	UpdateGeneratorsTableHeaders();
	UpdateGeneratorsTable();
	UpdateMaterialsLists();
	UpdateMaterialsSelection();
}

void CBondsGeneratorTab::UpdateGeneratorsTableHeaders() const
{
	CQtSignalBlocker blocker{ ui.tableClasses };
	ShowConvLabel(ui.tableClasses->horizontalHeaderItem(EColumn::MIN_DIST), "Min dist.", EUnitType::LENGTH);
	ShowConvLabel(ui.tableClasses->horizontalHeaderItem(EColumn::MAX_DIST), "Max dist.", EUnitType::LENGTH);
	ShowConvLabel(ui.tableClasses->horizontalHeaderItem(EColumn::DIAMETER), "Diameter", EUnitType::PARTICLE_DIAMETER);
}

void CBondsGeneratorTab::UpdateGeneratorsTable() const
{
	CQtSignalBlocker blocker{ ui.tableClasses };
	const auto oldPos = ui.tableClasses->CurrentCellPos();
	ui.tableClasses->setRowCount(0);

	for (int i = 0; i < static_cast<int>(m_generator->GeneratorsNumber()); ++i)
	{
		const SBondClass* bondClass = m_generator->Generator(i);
		ui.tableClasses->insertRow(i);

		ui.tableClasses->SetCheckBox(i,		   EColumn::ACTIVITY, bondClass->isActive);
		ui.tableClasses->SetItemEditable(i,	   EColumn::NAME,	  bondClass->name);
		ui.tableClasses->SetComboBox(i,		   EColumn::MATERIAL, m_pMaterialsDB->GetCompoundsNames(), m_pMaterialsDB->GetCompoundsKeys(), bondClass->compoundKey);
		ui.tableClasses->SetItemEditable(i,    EColumn::MIN_DIST, m_pUnitConverter->GetValue(EUnitType::LENGTH, bondClass->minDistance));
		ui.tableClasses->SetItemEditable(i,    EColumn::MAX_DIST, m_pUnitConverter->GetValue(EUnitType::LENGTH, bondClass->maxDistance));
		ui.tableClasses->SetItemEditable(i,    EColumn::DIAMETER, m_pUnitConverter->GetValue(EUnitType::PARTICLE_DIAMETER, bondClass->diameter));
		ui.tableClasses->SetCheckBox(i,		   EColumn::OVERLAY,  bondClass->isOverlayAllowed);
		ui.tableClasses->SetCheckBox(i,		   EColumn::MAT_SPEC, bondClass->isCompoundSpecific);
		ui.tableClasses->SetItemNotEditable(i, EColumn::NUMBER,   bondClass->generatedBonds);
		ui.tableClasses->SetProgressBar(i,     EColumn::PROGRESS, static_cast<int>(bondClass->completeness));
	}

	ui.tableClasses->RestoreSelectedCell(oldPos);
}

void CBondsGeneratorTab::UpdateMaterialsLists() const
{
	CQtSignalBlocker blocker{ ui.listMaterials1, ui.listMaterials2 };

	ui.listMaterials1->clear();
	ui.listMaterials2->clear();
	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->CompoundsNumber()); ++i)
	{
		ui.listMaterials1->InsertItemCheckable(i, m_pMaterialsDB->GetCompoundName(i));
		ui.listMaterials2->InsertItemCheckable(i, m_pMaterialsDB->GetCompoundName(i));
	}
}

void CBondsGeneratorTab::UpdateMaterialsSelection() const
{
	CQtSignalBlocker blocker{ ui.listMaterials1, ui.listMaterials2 };

	const SBondClass* currClass = m_generator->Generator(ui.tableClasses->currentRow());
	const bool active = currClass && currClass->isCompoundSpecific;
	ui.groupBoxMaterials->setEnabled(active);

	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->CompoundsNumber()); ++i)
	{
		ui.listMaterials1->SetItemChecked(i, active && VectorContains(currClass->compoundsLists.first,  m_pMaterialsDB->GetCompoundKey(i)));
		ui.listMaterials2->SetItemChecked(i, active && VectorContains(currClass->compoundsLists.second, m_pMaterialsDB->GetCompoundKey(i)));
	}
}

void CBondsGeneratorTab::AddClass() const
{
	SBondClass bondClass;
	bondClass.isActive = true;
	bondClass.name = GenerateClassName();
	m_generator->AddGenerator(bondClass);
	UpdateGeneratorsTable();
	ui.tableClasses->RestoreSelectedCell(static_cast<int>(m_generator->GeneratorsNumber()) - 1, EColumn::NAME);
}

void CBondsGeneratorTab::RemoveClass() const
{
	const auto oldPos = ui.tableClasses->CurrentCellPos();
	m_generator->RemoveGenerator(ui.tableClasses->currentRow());
	UpdateGeneratorsTable();
	ui.tableClasses->RestoreSelectedCell(oldPos);
	UpdateMaterialsSelection();
}

void CBondsGeneratorTab::UpClass() const
{
	const auto oldPos = ui.tableClasses->CurrentCellPos();
	m_generator->UpGenerator(ui.tableClasses->currentRow());
	UpdateGeneratorsTable();
	ui.tableClasses->RestoreSelectedCell(oldPos.first - 1, oldPos.second);
}

void CBondsGeneratorTab::DownClass() const
{
	const auto oldPos = ui.tableClasses->CurrentCellPos();
	m_generator->DownGenerator(ui.tableClasses->currentRow());
	UpdateGeneratorsTable();
	ui.tableClasses->RestoreSelectedCell(oldPos.first + 1, oldPos.second);
}

void CBondsGeneratorTab::BondClassChanged() const
{
	for (int i = 0; i < static_cast<int>(m_generator->GeneratorsNumber()); ++i)
	{
		SBondClass* bondClass = m_generator->Generator(i);

		bondClass->isActive           = ui.tableClasses->GetCheckBoxChecked(i, EColumn::ACTIVITY);
		bondClass->name               = ui.tableClasses->item(i, EColumn::NAME)->text().simplified().toStdString();
		bondClass->compoundKey        = ui.tableClasses->GetComboBoxValue(i, EColumn::MATERIAL).toString().toStdString();
		bondClass->minDistance        = GetConvValue(ui.tableClasses->item(i, EColumn::MIN_DIST), EUnitType::LENGTH);
		bondClass->maxDistance        = GetConvValue(ui.tableClasses->item(i, EColumn::MAX_DIST), EUnitType::LENGTH);
		bondClass->diameter           = std::max(GetConvValue(ui.tableClasses->item(i, EColumn::DIAMETER), EUnitType::PARTICLE_DIAMETER), 0.);
		bondClass->isOverlayAllowed   = ui.tableClasses->GetCheckBoxChecked(i, EColumn::OVERLAY);
		bondClass->isCompoundSpecific = ui.tableClasses->GetCheckBoxChecked(i, EColumn::MAT_SPEC);
	}

	UpdateGeneratorsTable();
	UpdateMaterialsSelection();
}

void CBondsGeneratorTab::BondClassSelected(int _currRow, int _currCol, int _prevRow, int _prevCol) const
{
	if (_currRow == _prevRow) return;
	UpdateMaterialsSelection();
}

void CBondsGeneratorTab::BondClassDeselected() const
{
	ui.tableClasses->clearSelection();
	ui.tableClasses->setCurrentItem(nullptr);
	UpdateMaterialsSelection();
}

void CBondsGeneratorTab::SelectedMaterialsChanged() const
{
	SBondClass* currClass = m_generator->Generator(ui.tableClasses->currentRow());
	if (!currClass) return;

	currClass->compoundsLists.first.clear();
	currClass->compoundsLists.second.clear();

	for (int i = 0; i < ui.listMaterials1->count(); ++i)
		if (ui.listMaterials1->GetItemChecked(i))
			currClass->compoundsLists.first.push_back(m_pMaterialsDB->GetCompoundKey(i));

	for (int i = 0; i < ui.listMaterials2->count(); ++i)
		if (ui.listMaterials2->GetItemChecked(i))
			currClass->compoundsLists.second.push_back(m_pMaterialsDB->GetCompoundKey(i));
}

void CBondsGeneratorTab::DeleteBondsClicked()
{
	if (QMessageBox::question(this, "Remove bonds", "Do you really want to remove all bonds?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	m_pSystemStructure->DeleteAllBonds();
	emit ObjectsChanged();
}

void CBondsGeneratorTab::StartStopClicked()
{
	if (m_generator->Status() == ERunningStatus::IDLE)
		StartGeneration();
	else if (m_generator->Status() == ERunningStatus::RUNNING)
		StopGeneration();
}

void CBondsGeneratorTab::StartGeneration()
{
	// reset statistics
	for (auto& generator : m_generator->Generators())
	{
		generator->generatedBonds = 0;
		generator->completeness   = 0;
	}

	// perform checks
	if (!m_generator->IsDataCorrect())
	{
		ui.statusMessage->setText(QString::fromStdString(m_generator->ErrorMessage()));
		return;
	}
	if (m_pSystemStructure->GetNumberOfSpecificObjects(SOLID_BOND) != 0 || m_pSystemStructure->GetNumberOfSpecificObjects(LIQUID_BOND) != 0)
		if (QMessageBox::warning(this, "Bonds generator", "The scene already contains bonds. Are you sure you want to start the generation?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes)
			return;

	// update GUI elements
	SetWindowModal(true);
	EnableControls(false);
	emit DisableOpenGLView();
	ui.statusMessage->setText("Generation started...");
	ui.buttonStartGeneration->setText("Stop");

	// start generation
	m_generatorThread.start();
	m_statisticsTimer.start(100);
}

void CBondsGeneratorTab::StopGeneration()
{
	// request stop
	m_generatorProcess.StopGeneration();
	// stop statistics timer
	m_statisticsTimer.stop();
	// wait until generation is finished
	while (m_generator->Status() != ERunningStatus::IDLE)
		m_generatorThread.wait(100);
}

void CBondsGeneratorTab::GenerationFinished()
{
	// stop generation
	m_statisticsTimer.stop();
	m_generatorThread.exit();

	// update GUI elements
	ui.statusMessage->setText("Generation finished");
	ui.buttonStartGeneration->setText("Generate");
	SetWindowModal(false);
	EnableControls(true);
	emit EnableOpenGLView();
	emit ObjectsChanged();
}

void CBondsGeneratorTab::UpdateGenerationStatistics() const
{
	CQtSignalBlocker blocker{ ui.tableClasses };
	for (int i = 0; i < static_cast<int>(m_generator->GeneratorsNumber()); ++i)
	{
		ui.tableClasses->item(i, EColumn::NUMBER)->setText(QString::number(m_generator->Generator(i)->generatedBonds));
		ui.tableClasses->SetProgressBar(i, EColumn::PROGRESS, static_cast<int>(m_generator->Generator(i)->completeness));
	}
}

void CBondsGeneratorTab::EnableControls(bool _enable) const
{
	ui.tableClasses->setEnabled(_enable);
	ui.buttonAddClass->setEnabled(_enable);
	ui.buttonRemoveClass->setEnabled(_enable);
	ui.buttonUpClass->setEnabled(_enable);
	ui.buttonDownClass->setEnabled(_enable);
	ui.buttonDeleteBonds->setEnabled(_enable);
	ui.groupBoxMaterials->setEnabled(_enable);
}

std::string CBondsGeneratorTab::GenerateClassName() const
{
	// gather existing names
	std::vector<std::string> names;
	for (const auto& g : m_generator->Generators())
		names.push_back(g->name);
	// find unique name
	for (size_t i = 0;; ++i)
	{
		const std::string name = "Class" + std::to_string(i);
		if (!VectorContains(names, name))
			return name;
	}
}
