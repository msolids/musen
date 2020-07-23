/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ObjectsGeneratorTab.h"

CObjectsGeneratorTab::CObjectsGeneratorTab( CGenerationManager* _pGenerationManager, QWidget *parent ) :
	CMusenDialog( parent ), m_pGenerationManager( _pGenerationManager )
{
	ui.setupUi(this);

	m_pAggloCompounds = new CAggloCompounds(this);
	m_bAvoidSignal = false;

	m_sHelpFileName = "Users Guide/Dynamic Generator.pdf";
	InitializeConnections();
}

void CObjectsGeneratorTab::InitializeConnections() const
{
	connect( ui.addGenerator, SIGNAL( clicked() ), this, SLOT( AddGenerator() ) );
	connect( ui.removeGenerator, SIGNAL( clicked() ), this, SLOT( DeleteGenerator() ) );
	connect( ui.generatorsList, SIGNAL( itemChanged( QListWidgetItem* ) ), this, SLOT( GeneratorWasChanged() ) );

	connect( ui.generatorsList, SIGNAL( currentItemChanged( QListWidgetItem*, QListWidgetItem* ) ), this, SLOT( UpdateSelectedGenerator() ) );
	connect(ui.radioParticles, SIGNAL(clicked()), this, SLOT(GeneratorTypeChanged()));
	connect(ui.agglomRadio, SIGNAL(clicked()), this, SLOT(GeneratorTypeChanged()));

	connect(ui.radioFixedVelocity, SIGNAL(clicked()), this, SLOT(VelocityTypeChanged()));
	connect(ui.radioRandomVelocity, SIGNAL(clicked()), this, SLOT(VelocityTypeChanged()));

	connect(ui.pushButtonAggloMaterials, SIGNAL(clicked()), m_pAggloCompounds, SLOT(show()));
	connect(this, &CObjectsGeneratorTab::PointersAreSet, this, &CObjectsGeneratorTab::SetupAggloCompounds);

	connect(ui.agglomerateCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(NewAgglomerateChosen(int)));

	connect(ui.generationVolume, SIGNAL(currentIndexChanged(int)), this, SLOT(SetParameters()));
	connect(ui.insideGeometriesCheckBox, SIGNAL(clicked()), this, SLOT(SetParameters()));
	connect(ui.lineEditVeloX, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.lineEditVeloY, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.lineEditVeloZ, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.lineEditMagnitude, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.mixtureCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(SetParameters()));
	connect(ui.scalingFact, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.startTime, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.generationRate, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.endTime, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
	connect(ui.updateStep, SIGNAL(editingFinished()), this, SLOT(SetParameters()));
}

void CObjectsGeneratorTab::DeleteGenerator()
{
	int nIndex = ui.generatorsList->currentRow();
	if ( nIndex < 0 ) return;
	m_pGenerationManager->DeleteGenerator( nIndex );
	UpdateWholeView();
}

void CObjectsGeneratorTab::AddGenerator()
{
	m_pGenerationManager->CreateNewGenerator();
	UpdateWholeView();
}

void CObjectsGeneratorTab::GeneratorTypeChanged()
{
	int nIndex = ui.generatorsList->currentRow();
	CObjectsGenerator* pGenerator = m_pGenerationManager->GetGenerator( nIndex );
	if (!pGenerator) return;
	pGenerator->m_bGenerateMixture = ui.radioParticles->isChecked();
	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::VelocityTypeChanged()
{
	int index = ui.generatorsList->currentRow();
	CObjectsGenerator* pGenerator = m_pGenerationManager->GetGenerator(index);
	if (!pGenerator) return;
	pGenerator->m_bRandomVelocity = ui.radioRandomVelocity->isChecked();
	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::GeneratorWasChanged()
{
	if (m_bAvoidSignal) return;
	for ( int i = 0; i < ui.generatorsList->count(); i++ )
	{
		QListWidgetItem *pItem = ui.generatorsList->item(i);
		QString sNewName = pItem->text().simplified();
		if (!sNewName.isEmpty())
		{
			m_pGenerationManager->GetGenerator(i)->m_sName = qs2ss(sNewName);
			m_pGenerationManager->GetGenerator(i)->m_bActive = (pItem->checkState() == Qt::Checked);
		}
	}
}

void CObjectsGeneratorTab::UpdateGeneratorsList()
{
	if (m_bAvoidSignal) return;
	bool bBlocked = ui.generatorsList->blockSignals(true);
	int iOld = ui.generatorsList->currentRow();
	ui.generatorsList->clear();
	for (unsigned i = 0; i < m_pGenerationManager->GetGeneratorsNumber(); ++i)
	{
		ui.generatorsList->insertItem(i, ss2qs(m_pGenerationManager->GetGenerator(i)->m_sName));
		ui.generatorsList->item(i)->setFlags(ui.generatorsList->item(i)->flags() | Qt::ItemIsEditable | Qt::ItemIsUserCheckable);
		ui.generatorsList->item(i)->setCheckState(m_pGenerationManager->GetGenerator(i)->m_bActive ? Qt::Checked : Qt::Unchecked);
	}
	if (ui.generatorsList->count() != 0)
		if (iOld == -1)
			ui.generatorsList->setCurrentRow(0);
		else if (iOld < ui.generatorsList->count())
			ui.generatorsList->setCurrentRow(iOld);
		else
			ui.generatorsList->setCurrentRow(ui.generatorsList->count() - 1);
	ui.generatorsList->blockSignals(bBlocked);
}

void CObjectsGeneratorTab::UpdateSelectedGenerator()
{
	int index = ui.generatorsList->currentRow();
	CObjectsGenerator* pGenerator = m_pGenerationManager->GetGenerator( index );
	bool bGenSelected = (pGenerator != nullptr);
	ui.frameSettings->setEnabled(bGenSelected);
	if (!bGenSelected) return;

	m_bAvoidSignal = true;

	// main parameters
	ui.generationVolume->clear();
	for (unsigned i = 0; i < m_pSystemStructure->GetAnalysisVolumesNumber(); ++i)
		ui.generationVolume->insertItem(i, ss2qs(m_pSystemStructure->GetAnalysisVolume(i)->sName));
	ui.generationVolume->setCurrentIndex(static_cast<int>(m_pSystemStructure->GetAnalysisVolumeIndex(pGenerator->m_sVolumeKey)));
	ui.insideGeometriesCheckBox->setChecked(pGenerator->m_bInsideGeometries);

	// Particles
	ui.mixtureCombo->clear();
	for (unsigned i = 0; i < m_pSystemStructure->m_MaterialDatabase.MixturesNumber(); ++i)
		ui.mixtureCombo->insertItem(i, ss2qs(m_pSystemStructure->m_MaterialDatabase.GetMixtureName(i)));
	ui.mixtureCombo->setCurrentIndex(m_pSystemStructure->m_MaterialDatabase.GetMixtureIndex(pGenerator->m_sMixtureKey));


	// Agglomerates
	ui.scalingFact->setText(QString::number(pGenerator->m_dAgglomerateScaleFactor));
	ui.agglomerateCombo->clear();
	for (unsigned i = 0; i < m_pAgglomDB->GetAgglomNumber(); ++i)
		ui.agglomerateCombo->insertItem(i, ss2qs(m_pAgglomDB->GetAgglomerate(i)->sName));
	ui.agglomerateCombo->setCurrentIndex(m_pAgglomDB->GetAgglomerateIndex(pGenerator->m_sAgglomerateKey));
	m_pAggloCompounds->SetGenerator(pGenerator);

	// Velocity
	ShowConvLabel(ui.labelVelocity, "X:Y:Z", EUnitType::VELOCITY);
	ShowConvValue(ui.lineEditVeloX, pGenerator->m_vObjInitVel.x, EUnitType::VELOCITY);
	ShowConvValue(ui.lineEditVeloY, pGenerator->m_vObjInitVel.y, EUnitType::VELOCITY);
	ShowConvValue(ui.lineEditVeloZ, pGenerator->m_vObjInitVel.z, EUnitType::VELOCITY);
	ShowConvLabel(ui.labelMagnitude, "Magnitude", EUnitType::VELOCITY);
	ShowConvValue(ui.lineEditMagnitude, pGenerator->m_dVelMagnitude, EUnitType::VELOCITY);

	// Generation rate
	ShowConvLabel( ui.startTimeLabel, "Start time", EUnitType::TIME );
	ShowConvLabel( ui.endTimeLabel, "End time", EUnitType::TIME );
	ui.labelGenRate->setText("Generation rate[1/" + ss2qs(m_pUnitConverter->GetSelectedUnit(EUnitType::TIME)) + "]");
	ShowConvLabel( ui.updateStepLabel, "Updating step", EUnitType::TIME );

	ShowConvValue( ui.startTime, pGenerator->m_dStartGenerationTime, EUnitType::TIME );
	ShowConvValue( ui.endTime, pGenerator->m_dEndGenerationTime, EUnitType::TIME );
	ui.generationRate->setText(QString::number(pGenerator->m_dGenerationRate / m_pUnitConverter->GetValue(EUnitType::TIME, 1)));
	ShowConvValue( ui.updateStep, pGenerator->m_dUpdateStep, EUnitType::TIME );

	// Radio buttons
	ui.radioParticles->setChecked(pGenerator->m_bGenerateMixture);
	ui.agglomRadio->setChecked(!pGenerator->m_bGenerateMixture);
	ui.frameAgglomerate->setEnabled(!pGenerator->m_bGenerateMixture);

	ui.radioFixedVelocity->setChecked(!pGenerator->m_bRandomVelocity);
	ui.frameFixed->setEnabled(!pGenerator->m_bRandomVelocity);
	ui.radioRandomVelocity->setChecked(pGenerator->m_bRandomVelocity);
	ui.frameRandom->setEnabled(pGenerator->m_bRandomVelocity);

	m_bAvoidSignal = false;
}

void CObjectsGeneratorTab::UpdateWholeView()
{
	UpdateGeneratorsList();
	UpdateSelectedGenerator();
}

void CObjectsGeneratorTab::SetParameters()
{
	if (m_bAvoidSignal) return;

	int nIndex = ui.generatorsList->currentRow();
	CObjectsGenerator* pGenerator = m_pGenerationManager->GetGenerator( nIndex );
	if (!pGenerator) return;

	// main parameters
	const CAnalysisVolume* pVolume = m_pSystemStructure->GetAnalysisVolume(ui.generationVolume->currentIndex());
	if (pVolume)
		pGenerator->m_sVolumeKey = pVolume->sKey;
	pGenerator->m_bInsideGeometries = ui.insideGeometriesCheckBox->isChecked();

	// Generate mixture
	pGenerator->m_bGenerateMixture = ui.radioParticles->isChecked();
	const CMixture* pMixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(ui.mixtureCombo->currentIndex());
	if (pMixture)
		pGenerator->m_sMixtureKey = pMixture->GetKey();

	// Agglomerates
	SAgglomerate* pAgglom = m_pAgglomDB->GetAgglomerate(ui.agglomerateCombo->currentIndex());
	if (pAgglom)
		pGenerator->m_sAgglomerateKey = pAgglom->sKey;
	pGenerator->m_dAgglomerateScaleFactor = ui.scalingFact->text().toDouble();
	if (pGenerator->m_dAgglomerateScaleFactor <= 0) pGenerator->m_dAgglomerateScaleFactor = 1;

	// Velocity
	pGenerator->m_bRandomVelocity = ui.radioRandomVelocity->isChecked();
	pGenerator->m_vObjInitVel.x = GetConvValue(ui.lineEditVeloX, EUnitType::VELOCITY);
	pGenerator->m_vObjInitVel.y = GetConvValue(ui.lineEditVeloY, EUnitType::VELOCITY);
	pGenerator->m_vObjInitVel.z = GetConvValue(ui.lineEditVeloZ, EUnitType::VELOCITY);
	pGenerator->m_dVelMagnitude = GetConvValue(ui.lineEditMagnitude, EUnitType::VELOCITY);
	if (pGenerator->m_dVelMagnitude < 0) pGenerator->m_dVelMagnitude = 0;

	// Generation rate
	pGenerator->m_dStartGenerationTime = GetConvValue(ui.startTime, EUnitType::TIME);
	if (pGenerator->m_dStartGenerationTime < 0) pGenerator->m_dStartGenerationTime = 0;
	pGenerator->m_dEndGenerationTime = GetConvValue(ui.endTime, EUnitType::TIME);
	pGenerator->m_dEndGenerationTime = std::max(pGenerator->m_dEndGenerationTime, pGenerator->m_dStartGenerationTime);
	pGenerator->m_dGenerationRate = ui.generationRate->text().toDouble() * m_pUnitConverter->GetValue(EUnitType::TIME, 1);
	if (pGenerator->m_dGenerationRate < 0) pGenerator->m_dGenerationRate = 0;
	pGenerator->m_dUpdateStep = GetConvValue(ui.updateStep, EUnitType::TIME);
	if (pGenerator->m_dUpdateStep <= 0) pGenerator->m_dUpdateStep = 1e-3;

	UpdateSelectedGenerator();
	emit UpdateOpenGLView();
}

void CObjectsGeneratorTab::SetupAggloCompounds()
{
	m_pAggloCompounds->SetPointers(m_pSystemStructure, m_pAgglomDB);
}

void CObjectsGeneratorTab::NewAgglomerateChosen(int _nIndex)
{
	if (m_bAvoidSignal) return;
	int nGenerIndex = ui.generatorsList->currentRow();
	CObjectsGenerator* pGenerator = m_pGenerationManager->GetGenerator(nGenerIndex);
	if (!pGenerator) return;
	SAgglomerate* pAgglom = m_pAgglomDB->GetAgglomerate(_nIndex);
	if (pAgglom)
		pGenerator->m_sAgglomerateKey = pAgglom->sKey;
}
