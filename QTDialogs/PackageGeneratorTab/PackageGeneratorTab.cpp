/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "PackageGeneratorTab.h"
#include "qtOperations.h"
#include <QProgressBar>
#include <QKeyEvent>
#include <QThread>
#include <QMessageBox>

#include "QtSignalBlocker.h"

CPackageGeneratorThread::CPackageGeneratorThread(CPackageGenerator* _pPackageGenerator, QObject *parent /*= 0*/) :QObject(parent)
{
	m_pPackageGenerator = _pPackageGenerator;
}

CPackageGeneratorThread::~CPackageGeneratorThread() {}

void CPackageGeneratorThread::StartGeneration()
{
	m_pPackageGenerator->StartGeneration();
	emit finished();
}

void CPackageGeneratorThread::StopGeneration()
{
	if (m_pPackageGenerator->Status() == ERunningStatus::RUNNING)
		m_pPackageGenerator->SetStatus(ERunningStatus::TO_BE_STOPPED);
}


CPackageGeneratorTab::CPackageGeneratorTab(CPackageGenerator* _pPackageGenerator, QWidget *parent) :CMusenDialog(parent)
{
	ui.setupUi(this);
	m_pPackageGenerator = _pPackageGenerator;

	// set table properties
	ui.generationTable->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	ui.generationTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	m_pPackageGeneratorThread = nullptr;
	m_pQTThread = nullptr;

	m_sHelpFileName = "Users Guide/Package generator.pdf";
	m_bGenerationStarted = false;
	InitializeConnections();
}

void CPackageGeneratorTab::InitializeConnections()
{
	// signals from the volume list
	connect(ui.generatorsList, SIGNAL(currentItemChanged(QListWidgetItem*, QListWidgetItem*)), this, SLOT(SelectedVolumeChanged()));
	connect(ui.generatorsList, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(GeneratorItemChanged(QListWidgetItem*)));

	// signals from buttons
	connect(ui.addGenerator, SIGNAL(clicked()), this, SLOT(AddGenerator()));
	connect(ui.removeGenerator, SIGNAL(clicked()), this, SLOT(DeleteGenerator()));
	connect(ui.buttonUp, SIGNAL(clicked()), this, SLOT(UpGenerator()));
	connect(ui.buttonDown, SIGNAL(clicked()), this, SLOT(DownGenerator()));

	// signals from the combo boxes
	connect(ui.volumeTypesCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(GenerationDataChanged()));
	connect(ui.mixtureCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(GenerationDataChanged()));
	connect(ui.targetPorosity,		&QLineEdit::editingFinished, this, &CPackageGeneratorTab::GenerationDataChanged);
	connect(ui.maxAllowedOverlap,	&QLineEdit::editingFinished, this, &CPackageGeneratorTab::GenerationDataChanged);
	connect(ui.maxIterNumber,		&QLineEdit::editingFinished, this, &CPackageGeneratorTab::GenerationDataChanged);
	connect(ui.lineEditInitVelX,	&QLineEdit::editingFinished, this, &CPackageGeneratorTab::GenerationDataChanged);
	connect(ui.lineEditInitVelY,	&QLineEdit::editingFinished, this, &CPackageGeneratorTab::GenerationDataChanged);
	connect(ui.lineEditInitVelZ,	&QLineEdit::editingFinished, this, &CPackageGeneratorTab::GenerationDataChanged);
	connect(ui.checkBoxInsideGeometry, SIGNAL(clicked()), this, SLOT(GenerationDataChanged()));

	connect(ui.comboSimulatorType, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &CPackageGeneratorTab::GeneratorTypeChanged);
	connect(ui.lineEditVerletCoeff, &QLineEdit::editingFinished, this, &CPackageGeneratorTab::GeneratorTypeChanged);

	// ======== GENERATION GROUP BOX
	// push_buttons
	connect(ui.startGenerationButton, SIGNAL(clicked()), this, SLOT(StartGeneration()));
	connect(ui.buttonDeleteParticles, SIGNAL(clicked()), this, SLOT(DeleteAllParticles()));

	// signals from timer
	connect( &m_UpdateTimer, SIGNAL(timeout()), this, SLOT( UpdateGenerationStatistics() ) );
}

void CPackageGeneratorTab::AddGenerator()
{
	m_pPackageGenerator->AddGenerator();
	UpdateGeneratorsList();
	UpdateSelectedGeneratorInfo();
	UpdateGeneratorsTable();
}

void CPackageGeneratorTab::UpGenerator()
{
	int oldRow = ui.generatorsList->currentRow();
	m_pPackageGenerator->UpGenerator(oldRow);
	ui.generatorsList->setCurrentRow(oldRow == 0 ? 0 : --oldRow);
	UpdateWholeView();
}

void CPackageGeneratorTab::DownGenerator()
{
	int oldRow = ui.generatorsList->currentRow();
	int lastRow = ui.generatorsList->count() - 1;
	m_pPackageGenerator->DownGenerator(oldRow);
	ui.generatorsList->setCurrentRow(oldRow == lastRow ? lastRow : ++oldRow);
	UpdateWholeView();

}

void CPackageGeneratorTab::keyPressEvent(QKeyEvent *event)
{
	switch (((QKeyEvent*)event)->key())
	{
	case Qt::Key_Escape:
		QDialog::keyPressEvent(event);
		break;
	case Qt::Key_Return: break;
	default: CMusenDialog::keyPressEvent(event);
	}
}

void CPackageGeneratorTab::UpdateGeneratorsList()
{
	bool bSignalsFlag = ui.generatorsList->blockSignals(true);

	int nOldRow = ui.generatorsList->currentRow();
	ui.generatorsList->clear();
	for (int i = 0; i < (int)m_pPackageGenerator->GeneratorsNumber(); ++i)
	{
		const SPackage* pGen = m_pPackageGenerator->Generator(i);
		ui.generatorsList->insertItem(i, ss2qs(pGen->name));
		ui.generatorsList->item(i)->setFlags(ui.generatorsList->item(i)->flags() | Qt::ItemIsEditable | Qt::ItemIsUserCheckable);
		ui.generatorsList->item(i)->setCheckState(pGen->active ? Qt::Checked : Qt::Unchecked);
	}

	if ((nOldRow >= 0) && (nOldRow < (int)m_pPackageGenerator->GeneratorsNumber()))
		ui.generatorsList->setCurrentRow(nOldRow);
	else if (nOldRow != -1)
		ui.generatorsList->setCurrentRow(ui.generatorsList->count() - 1);
	else
		ui.generatorsList->setCurrentRow(0);
	ui.generatorsList->blockSignals(bSignalsFlag);

	UpdateSelectedGeneratorInfo();
	UpdateGeneratorsTable();
}

void CPackageGeneratorTab::UpdateSelectedGeneratorInfo()
{
	int nSelectedRow = ui.generatorsList->currentRow();
	bool bGeneratorSelected = true;
	if ((nSelectedRow < 0) || (nSelectedRow >= (int)m_pPackageGenerator->GeneratorsNumber()))
		bGeneratorSelected = false;

	ui.volumeTypesCombo->setEnabled(bGeneratorSelected);
	ui.mixtureCombo->setEnabled(bGeneratorSelected);
	ui.maxAllowedOverlap->setEnabled(bGeneratorSelected);
	ui.targetPorosity->setEnabled(bGeneratorSelected);
	ui.maxIterNumber->setEnabled(bGeneratorSelected);
	ui.lineEditInitVelX->setEnabled(bGeneratorSelected);
	ui.lineEditInitVelY->setEnabled(bGeneratorSelected);
	ui.lineEditInitVelZ->setEnabled(bGeneratorSelected);
	ui.checkBoxInsideGeometry->setEnabled(bGeneratorSelected);
	if (!bGeneratorSelected) return;

	m_bAvoidSignal = true;

	const SPackage* pGen = m_pPackageGenerator->Generator(nSelectedRow);
	ui.volumeTypesCombo->setCurrentIndex(static_cast<int>(m_pSystemStructure->GetAnalysisVolumeIndex(pGen->volumeKey)));

	// set correct index of material combo box
	ui.mixtureCombo->setCurrentIndex(m_pMaterialsDB->GetMixtureIndex(pGen->mixtureKey));

	ui.targetPorosity->setText(QString::number(pGen->targetPorosity));
	ShowConvValue(ui.maxAllowedOverlap, pGen->targetMaxOverlap, EUnitType::PARTICLE_DIAMETER);
	ui.maxIterNumber->setText(QString::number(pGen->maxIterations));
	ShowConvValue(ui.lineEditInitVelX, ui.lineEditInitVelY, ui.lineEditInitVelZ, pGen->initVelocity, EUnitType::VELOCITY);
	ui.checkBoxInsideGeometry->setChecked(pGen->insideGeometry);
	m_bAvoidSignal = false;
}

void CPackageGeneratorTab::UpdateGeneratorsTable()
{
	bool bSignalsFlag = ui.generationTable->blockSignals(true);

	// create table
	ui.generationTable->setRowCount(0);
	for (unsigned i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
	{
		ui.generationTable->insertRow(i);

		for (int j = 0; j < ui.generationTable->columnCount(); ++j)
			ui.generationTable->setItem(i, j, new QTableWidgetItem(""));
		ui.generationTable->item(i, VOLUME_NAME)->setFlags(ui.generationTable->item(i, VOLUME_NAME)->flags() ^ Qt::ItemIsEditable); // disable editing of name
		ui.generationTable->item(i, PARTICLES)->setFlags(ui.generationTable->item(i, PARTICLES)->flags() ^ Qt::ItemIsEditable); // disable editing of particle
		ui.generationTable->item(i, MAX_OVERLAP)->setFlags(ui.generationTable->item(i, MAX_OVERLAP)->flags() ^ Qt::ItemIsEditable); // disable editing of max overlap
		ui.generationTable->item(i, AVER_OVERLAP)->setFlags(ui.generationTable->item(i, AVER_OVERLAP)->flags() ^ Qt::ItemIsEditable); // disable editing of average overlap

		// create progress bar with completeness level
		QProgressBar* pProgressBar = new QProgressBar();
		pProgressBar->setRange(0, 100);
		ui.generationTable->setCellWidget(i, COMPLETNESS, pProgressBar);
	}

	// fill table
	for (unsigned i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
	{
		const SPackage* pGen = m_pPackageGenerator->Generator(i);
		ui.generationTable->item(i, VOLUME_NAME)->setText(ss2qs(pGen->name));
		ShowParticlesNumberInTable(i);
		ShowConvValue(ui.generationTable->item(i, MAX_OVERLAP), pGen->maxReachedOverlap, EUnitType::PARTICLE_DIAMETER);// max reached overlap
		ShowConvValue(ui.generationTable->item(i, AVER_OVERLAP), pGen->avrReachedOverlap, EUnitType::PARTICLE_DIAMETER);// average reached overlap
		((QProgressBar*)(ui.generationTable->cellWidget(i, COMPLETNESS)))->setValue(pGen->completness);
	}

	// set rows color according to the activity
	for (unsigned i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
	{
		Qt::GlobalColor color = m_pPackageGenerator->Generator(i)->active ? Qt::white : Qt::gray;
		for (int j = 0; j < ui.generationTable->columnCount(); ++j)
			ui.generationTable->item(i, j)->setBackground(color);
	}

	ui.generationTable->blockSignals(bSignalsFlag);
}

void CPackageGeneratorTab::UpdateVolumesCombo()
{
	bool bSignalsFlag = ui.volumeTypesCombo->blockSignals(true);
	ui.volumeTypesCombo->clear();
	for (unsigned i = 0; i < m_pSystemStructure->GetAnalysisVolumesNumber(); ++i)
		ui.volumeTypesCombo->insertItem(i, ss2qs(m_pSystemStructure->GetAnalysisVolume(i)->sName));
	ui.volumeTypesCombo->setCurrentIndex(-1);
	ui.volumeTypesCombo->blockSignals(bSignalsFlag);
}

void CPackageGeneratorTab::UpdateMixturesCombo()
{
	bool bSignalsFlag = ui.mixtureCombo->blockSignals(true);
	ui.mixtureCombo->clear();
	for (unsigned j = 0; j < m_pMaterialsDB->MixturesNumber(); ++j)
		ui.mixtureCombo->insertItem(j, ss2qs(m_pMaterialsDB->GetMixtureName(j)));
	ui.mixtureCombo->setCurrentIndex(-1);
	ui.mixtureCombo->blockSignals(bSignalsFlag);
}

void CPackageGeneratorTab::UpdateUnitsInLabels()
{
	ShowConvLabel(ui.generationTable->horizontalHeaderItem(AVER_OVERLAP), "Aver overlap", EUnitType::PARTICLE_DIAMETER);
	ShowConvLabel(ui.generationTable->horizontalHeaderItem(MAX_OVERLAP), "Max overlap", EUnitType::PARTICLE_DIAMETER);
	ShowConvLabel(ui.labelMaxOverlap, "Max overlap", EUnitType::PARTICLE_DIAMETER);
	ShowConvLabel(ui.labelInitVelocity, "Initial velocity", EUnitType::VELOCITY);
}

void CPackageGeneratorTab::UpdateSimulatorType() const
{
	CQtSignalBlocker blocker{ ui.comboSimulatorType, ui.lineEditVerletCoeff };
	ui.comboSimulatorType->clear();
	ui.comboSimulatorType->insertItem(static_cast<int>(ESimulatorType::CPU), "CPU");
	ui.comboSimulatorType->insertItem(static_cast<int>(ESimulatorType::GPU), "GPU");
	ui.comboSimulatorType->setCurrentIndex(static_cast<int>(m_pPackageGenerator->GetSimulatorType()) - 1);
	ui.lineEditVerletCoeff->setText(QString::number(m_pPackageGenerator->GetVerletCoefficient()));
}

void CPackageGeneratorTab::ShowParticlesNumberInTable(unsigned _iGenerator)
{
	const SPackage* pGen = m_pPackageGenerator->Generator(_iGenerator);
	QFont font = ui.generationTable->item(_iGenerator, PARTICLES)->font();

	if (m_pPackageGenerator->Status() == ERunningStatus::RUNNING)
	{
		// generated particles
		font.setItalic(false);
		ui.generationTable->item(_iGenerator, PARTICLES)->setFont(font);
		ui.generationTable->item(_iGenerator, PARTICLES)->setForeground(QBrush(Qt::black));
		ui.generationTable->item(_iGenerator, PARTICLES)->setText(QString::number(pGen->generatedParticles));
	}
	else
	{
		// approximate number of particles to be generated
		font.setItalic(true);
		ui.generationTable->item(_iGenerator, PARTICLES)->setFont(font);
		ui.generationTable->item(_iGenerator, PARTICLES)->setForeground(QBrush(Qt::blue));
		ui.generationTable->item(_iGenerator, PARTICLES)->setText(QString::number(m_pPackageGenerator->ParticlesToGenerate(_iGenerator)));

	}
}

void CPackageGeneratorTab::GeneratorItemChanged(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	bool bSignalsFlag = ui.generatorsList->blockSignals(true);

	int nRow = ui.generatorsList->row(_pItem);
	QString sNewName = _pItem->text().simplified();
	SPackage* pGen = m_pPackageGenerator->Generator(nRow);
	if (!sNewName.isEmpty())
		pGen->name = qs2ss(sNewName);
	pGen->active = (_pItem->checkState() == Qt::Checked);
	_pItem->setText(ss2qs(pGen->name));

	ui.generatorsList->blockSignals(bSignalsFlag);

	UpdateGeneratorsTable();
}

void CPackageGeneratorTab::DeleteGenerator() // additionally generation class is removed
{
	int index = ui.generatorsList->currentRow();
	if (index < 0) return;
	m_pPackageGenerator->RemoveGenerator((unsigned)index);
	UpdateGeneratorsList();
	UpdateSelectedGeneratorInfo();
	UpdateGeneratorsTable();
}

void CPackageGeneratorTab::SelectedVolumeChanged()
{
	UpdateSelectedGeneratorInfo();
}

void CPackageGeneratorTab::UpdateWholeView()
{
	UpdateUnitsInLabels();
	UpdateMixturesCombo();
	UpdateVolumesCombo();
	UpdateGeneratorsList();
	UpdateSelectedGeneratorInfo();
	UpdateGeneratorsTable();
	UpdateSimulatorType();
}

void CPackageGeneratorTab::GenerationDataChanged()
{
	if (m_bAvoidSignal) return;
	int iGenerator = ui.generatorsList->currentRow();
	if (iGenerator < 0) return;

	SPackage* pGen = m_pPackageGenerator->Generator(iGenerator);

	int iVolume = ui.volumeTypesCombo->currentIndex();
	if (iVolume >= 0)
		pGen->volumeKey = m_pSystemStructure->GetAnalysisVolume(iVolume)->sKey;

	const CMixture* pMixture = m_pMaterialsDB->GetMixture(ui.mixtureCombo->currentIndex());
	pGen->mixtureKey = pMixture ? pMixture->GetKey() : "";

	pGen->targetPorosity = ui.targetPorosity->text().toDouble();
	pGen->targetMaxOverlap = GetConvValue(ui.maxAllowedOverlap, EUnitType::PARTICLE_DIAMETER);
	pGen->maxIterations = ui.maxIterNumber->text().toInt();
	pGen->initVelocity = GetConvValue(ui.lineEditInitVelX, ui.lineEditInitVelY, ui.lineEditInitVelZ, EUnitType::VELOCITY);
	pGen->insideGeometry = ui.checkBoxInsideGeometry->isChecked();

	// data correctness validation
	if (pGen->targetPorosity > 1) pGen->targetPorosity = 1;
	if (pGen->targetPorosity < 0) pGen->targetPorosity = 0;
	if (pGen->targetMaxOverlap < 0) pGen->targetMaxOverlap = 0;

	// particles to be generated
	ShowParticlesNumberInTable(iGenerator);
}

void CPackageGeneratorTab::GeneratorTypeChanged() const
{
	const auto type = static_cast<ESimulatorType>(ui.comboSimulatorType->currentIndex() + 1);
	m_pPackageGenerator->SetSimulatorType(type);
	m_pPackageGenerator->SetVerletCoefficient(ui.lineEditVerletCoeff->text().toDouble());
}

void CPackageGeneratorTab::StartGeneration()
{
	if ( m_bGenerationStarted == true ) // stop generation
	{
		m_pPackageGeneratorThread->StopGeneration();

		//! disable old timer to avoid updating from NULL class
		m_UpdateTimer.stop();

		while (m_pPackageGenerator->Status() != ERunningStatus::IDLE)
			m_pQTThread->wait( 100 );
		GenerationFinished();
	}
	else // start generation
	{
		// reset all statistics
		for (size_t i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
		{
			SPackage* pGen = m_pPackageGenerator->Generator(i);
			pGen->completness = 0;
			pGen->generatedParticles = 0;
			pGen->maxReachedOverlap = 0;
			pGen->avrReachedOverlap = 0;

			if ((pGen->active) && (!m_pSystemStructure->GetParticleIndicesInVolume(0, pGen->volumeKey, false).empty()))
					if (QMessageBox::warning(this, "Confirmation", tr("Scene already contains particles placed in the volume '%1'. New generation into the same volumes can lead to uncertainties. Continue generation?").arg(ss2qs(m_pSystemStructure->GetAnalysisVolume(pGen->volumeKey)->sName)), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes)
						return;
		}

		if (!m_pPackageGenerator->IsDataCorrect())
		{
			ui.errorMessage->setStyleSheet("QLabel { color : red; }");
			ui.errorMessage->setText(QString::fromStdString(m_pPackageGenerator->ErrorMessage()));
			return;
		}

		m_pPackageGeneratorThread = new CPackageGeneratorThread(m_pPackageGenerator);

		SetWindowModal(true);
		ui.errorMessage->setStyleSheet("QLabel { color : black; }");
		ui.errorMessage->setText( "Generation started..." );
		EnableControls(false);
		m_pQTThread = new QThread();
		connect( m_pQTThread, SIGNAL(started()), m_pPackageGeneratorThread, SLOT(StartGeneration()));
		connect( m_pPackageGeneratorThread, SIGNAL(finished()), this, SLOT(GenerationFinished()));
		m_pPackageGeneratorThread->moveToThread( m_pQTThread );
		m_pQTThread->start();

		ui.startGenerationButton->setText( "Stop" );
		// start update timer
		m_UpdateTimer.start(100);
		m_bGenerationStarted = true;
	}
}

void CPackageGeneratorTab::EnableControls(bool _bEnable)
{
	ui.volumeTypesCombo->setEnabled(_bEnable);
	ui.mixtureCombo->setEnabled(_bEnable);
	ui.targetPorosity->setEnabled(_bEnable);
	ui.lineEditInitVelX->setEnabled(_bEnable);
	ui.lineEditInitVelY->setEnabled(_bEnable);
	ui.lineEditInitVelZ->setEnabled(_bEnable);
	ui.frameGeneratorsList->setEnabled(_bEnable);
	ui.buttonDeleteParticles->setEnabled(_bEnable);
	ui.comboSimulatorType->setEnabled(_bEnable);
	ui.lineEditVerletCoeff->setEnabled(_bEnable);
}

void CPackageGeneratorTab::GenerationFinished()
{
	m_UpdateTimer.stop();
	if (m_pQTThread)
	{
		m_pQTThread->exit();
		m_pQTThread = nullptr;
		delete m_pPackageGeneratorThread;
		m_pPackageGeneratorThread = nullptr;
	}

	ui.errorMessage->setText( "Generation finished" );
	ui.startGenerationButton->setText( "Start" );
	EnableControls(true);
	m_bGenerationStarted = false;
	SetWindowModal(false);
	emit OpenGLViewShouldBeCentrated();
	emit ObjectsChanged();
}

void CPackageGeneratorTab::UpdateGenerationStatistics()
{
	m_bAvoidSignal = true;
	for (unsigned i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
	{
		const SPackage* pGen = m_pPackageGenerator->Generator(i);
		ShowParticlesNumberInTable(i); // generated particles
		ui.generationTable->item(i, MAX_OVERLAP)->setText(QString::number(m_pUnitConverter->GetValue(EUnitType::PARTICLE_DIAMETER, pGen->maxReachedOverlap))); // max reached overlap
		ShowConvValue(ui.generationTable->item(i, MAX_OVERLAP), pGen->maxReachedOverlap, EUnitType::PARTICLE_DIAMETER);  // max reached overlap
		ShowConvValue(ui.generationTable->item(i, AVER_OVERLAP), pGen->avrReachedOverlap, EUnitType::PARTICLE_DIAMETER); // average reached overlap
		((QProgressBar*)(ui.generationTable->cellWidget(i, COMPLETNESS)))->setValue(pGen->completness); // completeness
	}
	m_bAvoidSignal = false;
}

void CPackageGeneratorTab::DeleteAllParticles()
{
	m_pSystemStructure->DeleteAllParticles();
	emit ObjectsChanged();
}
