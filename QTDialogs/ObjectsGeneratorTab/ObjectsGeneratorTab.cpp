/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ObjectsGeneratorTab.h"
#include "QtSignalBlocker.h"

CObjectsGeneratorTab::CObjectsGeneratorTab(CGenerationManager* _generationManager, QWidget* _parent) :
	CMusenDialog{ _parent },
	m_generationManager{ _generationManager }
{
	ui.setupUi(this);

	m_sHelpFileName = "Users Guide/Dynamic Generator.pdf";
	InitializeConnections();
}

void CObjectsGeneratorTab::InitializeConnections() const
{
	connect(ui.addGenerator,    &QPushButton::clicked, this, &CObjectsGeneratorTab::AddGenerator);
	connect(ui.removeGenerator, &QPushButton::clicked, this, &CObjectsGeneratorTab::DeleteGenerator);

	connect(ui.generatorsList, &QListWidget::itemChanged,        this, &CObjectsGeneratorTab::GeneratorChanged);
	connect(ui.generatorsList, &QListWidget::currentItemChanged, this, &CObjectsGeneratorTab::UpdateSelectedGenerator);

	connect(ui.radioParticles, &QRadioButton::clicked, this, &CObjectsGeneratorTab::GeneratorTypeChanged);
	connect(ui.agglomRadio,    &QRadioButton::clicked, this, &CObjectsGeneratorTab::GeneratorTypeChanged);

	connect(ui.radioFixedVelocity,  &QRadioButton::clicked, this, &CObjectsGeneratorTab::VelocityTypeChanged);
	connect(ui.radioRandomVelocity, &QRadioButton::clicked, this, &CObjectsGeneratorTab::VelocityTypeChanged);

	connect(ui.pushButtonAggloMaterials, &QPushButton::clicked, &m_aggloCompounds, &CAggloCompounds::show);
	connect(this, &CObjectsGeneratorTab::PointersAreSet, this, &CObjectsGeneratorTab::SetupAggloCompounds);
	connect(ui.agglomerateCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CObjectsGeneratorTab::NewAgglomerateChosen);

	connect(ui.generationVolume,         QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.insideGeometriesCheckBox, &QCheckBox::clicked,                                 this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.lineEditMaxIterations,    &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.lineEditVeloX,            &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.lineEditVeloY,            &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.lineEditVeloZ,            &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.lineEditMagnitude,        &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.mixtureCombo,             QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.scalingFact,              &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.startTime,                &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.endTime,                  &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.updateStep,               &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.comboRateType,            QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CObjectsGeneratorTab::SetParameters);
	connect(ui.lineEditRateValue,        &QLineEdit::editingFinished,                         this, &CObjectsGeneratorTab::SetParameters);
}

void CObjectsGeneratorTab::AddGenerator()
{
	m_generationManager->CreateNewGenerator();
	UpdateWholeView();
}

void CObjectsGeneratorTab::DeleteGenerator()
{
	const int index = ui.generatorsList->currentRow();
	if (index < 0) return;
	m_generationManager->DeleteGenerator(index);
	UpdateWholeView();
}

void CObjectsGeneratorTab::GeneratorChanged() const
{
	for (int i = 0; i < ui.generatorsList->count(); ++i)
	{
		QListWidgetItem *item = ui.generatorsList->item(i);
		const std::string name = item->text().simplified().toStdString();
		if (!name.empty())
			m_generationManager->GetGenerator(i)->m_sName = name;
		m_generationManager->GetGenerator(i)->m_bActive = item->checkState() == Qt::Checked;
	}
}

void CObjectsGeneratorTab::GeneratorTypeChanged()
{
	const int index = ui.generatorsList->currentRow();
	CObjectsGenerator* generator = m_generationManager->GetGenerator(index);
	if (!generator) return;
	generator->m_bGenerateMixture = ui.radioParticles->isChecked();
	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::VelocityTypeChanged()
{
	const int index = ui.generatorsList->currentRow();
	CObjectsGenerator* generator = m_generationManager->GetGenerator(index);
	if (!generator) return;
	generator->m_bRandomVelocity = ui.radioRandomVelocity->isChecked();
	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::UpdateWholeView()
{
	UpdateGeneratorsList();
	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::UpdateGeneratorsList() const
{
	CQtSignalBlocker blocker{ ui.generatorsList };
	const int old = ui.generatorsList->currentRow();
	ui.generatorsList->clear();
	for (size_t i = 0; i < m_generationManager->GetGeneratorsNumber(); ++i)
	{
		ui.generatorsList->insertItem(static_cast<int>(i), QString::fromStdString(m_generationManager->GetGenerator(i)->m_sName));
		ui.generatorsList->item(static_cast<int>(i))->setFlags(ui.generatorsList->item(static_cast<int>(i))->flags() | Qt::ItemIsEditable | Qt::ItemIsUserCheckable);
		ui.generatorsList->item(static_cast<int>(i))->setCheckState(m_generationManager->GetGenerator(i)->m_bActive ? Qt::Checked : Qt::Unchecked);
	}
	if (ui.generatorsList->count() != 0)
	{
		if (old == -1)
			ui.generatorsList->setCurrentRow(0);
		else if (old < ui.generatorsList->count())
			ui.generatorsList->setCurrentRow(old);
		else
			ui.generatorsList->setCurrentRow(ui.generatorsList->count() - 1);
	}
}

void CObjectsGeneratorTab::UpdateSelectedGenerator()
{
	CObjectsGenerator* generator = m_generationManager->GetGenerator(ui.generatorsList->currentRow());
	ui.frameSettings->setEnabled(generator);
	if (!generator) return;

	CQtSignalBlocker blocker{ ui.generationVolume, ui.lineEditMaxIterations, ui.insideGeometriesCheckBox, ui.mixtureCombo, ui.radioParticles,
		ui.agglomerateCombo, ui.scalingFact, &m_aggloCompounds, ui.agglomRadio, ui.frameAgglomerate,
		ui.lineEditVeloX, ui.lineEditVeloY, ui.lineEditVeloZ, ui.lineEditMagnitude, ui.radioFixedVelocity, ui.radioRandomVelocity,
		ui.frameFixed, ui.frameRandom, ui.comboRateType, ui.startTime, ui.endTime, ui.updateStep, ui.lineEditRateValue };

	// main parameters
	ui.generationVolume->clear();
	for (size_t i = 0; i < m_pSystemStructure->AnalysisVolumesNumber(); ++i)
		ui.generationVolume->addItem(QString::fromStdString(m_pSystemStructure->AnalysisVolume(i)->Name()));
	ui.generationVolume->setCurrentIndex(static_cast<int>(m_pSystemStructure->AnalysisVolumeIndex(generator->m_sVolumeKey)));
	ui.lineEditMaxIterations->setText(QString::number(generator->m_maxIterations));
	ui.insideGeometriesCheckBox->setChecked(generator->m_bInsideGeometries);

	// objects - mixture
	ui.mixtureCombo->clear();
	for (size_t i = 0; i < m_pSystemStructure->m_MaterialDatabase.MixturesNumber(); ++i)
		ui.mixtureCombo->addItem(QString::fromStdString(m_pSystemStructure->m_MaterialDatabase.GetMixtureName(i)));
	ui.mixtureCombo->setCurrentIndex(m_pSystemStructure->m_MaterialDatabase.GetMixtureIndex(generator->m_sMixtureKey));
	ui.radioParticles->setChecked(generator->m_bGenerateMixture);

	// objects - agglomerates
	ui.agglomerateCombo->clear();
	for (size_t i = 0; i < m_pAgglomDB->GetAgglomNumber(); ++i)
		ui.agglomerateCombo->addItem(QString::fromStdString(m_pAgglomDB->GetAgglomerate(i)->sName));
	ui.agglomerateCombo->setCurrentIndex(m_pAgglomDB->GetAgglomerateIndex(generator->m_sAgglomerateKey));
	ui.scalingFact->setText(QString::number(generator->m_dAgglomerateScaleFactor));
	m_aggloCompounds.SetGenerator(generator);
	ui.agglomRadio->setChecked(!generator->m_bGenerateMixture);
	ui.frameAgglomerate->setEnabled(!generator->m_bGenerateMixture);

	// velocity
	ShowConvLabel(ui.labelVelocity, "X:Y:Z", EUnitType::VELOCITY);
	ShowConvLabel(ui.labelMagnitude, "Magnitude", EUnitType::VELOCITY);
	ShowConvValue(ui.lineEditVeloX, ui.lineEditVeloY, ui.lineEditVeloZ, generator->m_vObjInitVel, EUnitType::VELOCITY);
	ShowConvValue(ui.lineEditMagnitude, generator->m_dVelMagnitude, EUnitType::VELOCITY);
	ui.radioFixedVelocity->setChecked(!generator->m_bRandomVelocity);
	ui.radioRandomVelocity->setChecked(generator->m_bRandomVelocity);
	ui.frameFixed->setEnabled(!generator->m_bRandomVelocity);
	ui.frameRandom->setEnabled(generator->m_bRandomVelocity);

	// generation rate
	ShowConvLabel(ui.startTimeLabel,  "Start time",    EUnitType::TIME);
	ShowConvLabel(ui.endTimeLabel,    "End time",      EUnitType::TIME);
	ShowConvLabel(ui.updateStepLabel, "Updating step", EUnitType::TIME);
	ui.comboRateType->clear();
	ui.comboRateType->addItem("Generation rate [1/" + QString::fromStdString(m_pUnitConverter->GetSelectedUnit(EUnitType::TIME)) + "]", E2I(CObjectsGenerator::ERateType::GENERATION_RATE));
	ui.comboRateType->addItem("Objects per step", E2I(CObjectsGenerator::ERateType::OBJECTS_PER_STEP));
	ui.comboRateType->addItem("Objects total", E2I(CObjectsGenerator::ERateType::OBJECTS_TOTAL));
	ui.comboRateType->setCurrentIndex(E2I(generator->m_rateType));
	ShowConvValue(ui.startTime,  generator->m_dStartGenerationTime, EUnitType::TIME);
	ShowConvValue(ui.endTime,    generator->m_dEndGenerationTime,   EUnitType::TIME);
	ShowConvValue(ui.updateStep, generator->m_dUpdateStep,          EUnitType::TIME);
	if (generator->m_rateType == CObjectsGenerator::ERateType::GENERATION_RATE)
		ui.lineEditRateValue->setText(QString::number(generator->m_rateValue / m_pUnitConverter->GetValue(EUnitType::TIME, 1)));
	else
		ui.lineEditRateValue->setText(QString::number(generator->m_rateValue));
}

void CObjectsGeneratorTab::SetParameters()
{
	CObjectsGenerator* generator = m_generationManager->GetGenerator(ui.generatorsList->currentRow());
	if (!generator) return;

	// main parameters
	if (const CAnalysisVolume* volume = m_pSystemStructure->AnalysisVolume(ui.generationVolume->currentIndex()))
		generator->m_sVolumeKey = volume->Key();
	generator->m_maxIterations = ui.lineEditMaxIterations->text().toUInt();
	generator->m_bInsideGeometries = ui.insideGeometriesCheckBox->isChecked();

	// objects - mixture
	generator->m_bGenerateMixture = ui.radioParticles->isChecked();
	if (const CMixture* mixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(ui.mixtureCombo->currentIndex()))
		generator->m_sMixtureKey = mixture->GetKey();

	// objects - agglomerates
	if (const SAgglomerate* agglomerate = m_pAgglomDB->GetAgglomerate(ui.agglomerateCombo->currentIndex()))
		generator->m_sAgglomerateKey = agglomerate->sKey;
	generator->m_dAgglomerateScaleFactor = ui.scalingFact->text().toDouble() > 0.0 ? ui.scalingFact->text().toDouble() : 1.0;

	// velocity
	generator->m_bRandomVelocity = ui.radioRandomVelocity->isChecked();
	generator->m_vObjInitVel = GetConvValue(ui.lineEditVeloX, ui.lineEditVeloY, ui.lineEditVeloZ, EUnitType::VELOCITY);
	generator->m_dVelMagnitude = std::max(GetConvValue(ui.lineEditMagnitude, EUnitType::VELOCITY), 0.0);

	// generation rate
	generator->m_dStartGenerationTime = std::max(GetConvValue(ui.startTime, EUnitType::TIME), 0.0);
	generator->m_dEndGenerationTime = std::max(GetConvValue(ui.endTime, EUnitType::TIME), generator->m_dStartGenerationTime);
	generator->m_dUpdateStep = GetConvValue(ui.updateStep, EUnitType::TIME) > 0.0 ? GetConvValue(ui.updateStep, EUnitType::TIME) : 1e-3;
	generator->m_rateType = static_cast<CObjectsGenerator::ERateType>(ui.comboRateType->currentData().toUInt());
	if (generator->m_rateType == CObjectsGenerator::ERateType::GENERATION_RATE)
		generator->m_rateValue = std::max(ui.lineEditRateValue->text().toDouble() * m_pUnitConverter->GetValue(EUnitType::TIME, 1), 0.0);
	else
		generator->m_rateValue = std::max(ui.lineEditRateValue->text().toDouble(), 0.0);

	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::SetupAggloCompounds()
{
	m_aggloCompounds.SetPointers(m_pSystemStructure, m_pAgglomDB);
}

void CObjectsGeneratorTab::NewAgglomerateChosen(int _index) const
{
	const int iGenerator = ui.generatorsList->currentRow();
	CObjectsGenerator* generator = m_generationManager->GetGenerator(iGenerator);
	if (!generator) return;
	SAgglomerate* agglom = m_pAgglomDB->GetAgglomerate(_index);
	if (!agglom) return;
	generator->m_sAgglomerateKey = agglom->sKey;
}
