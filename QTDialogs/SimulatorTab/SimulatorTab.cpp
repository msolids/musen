/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SimulatorTab.h"
#include "SelectiveSavingTab.h"
#include <QStandardItemModel>
#include <QMessageBox>
#include <QThread>

void CSimulatorThread::StartSimulation()
{
	m_pSimulator->Simulate();
	emit finished();
}

void CSimulatorThread::StopSimulation()
{
	if (m_pSimulator->GetCurrentStatus() != ERunningStatus::IDLE)
		m_pSimulator->SetCurrentStatus(ERunningStatus::TO_BE_STOPPED);
}

void CSimulatorThread::PauseSimulation()
{
	if (m_pSimulator->GetCurrentStatus() == ERunningStatus::RUNNING)
		m_pSimulator->SetCurrentStatus(ERunningStatus::TO_BE_PAUSED);
}


CSimulatorTab::CSimulatorTab(CSimulatorManager* _pSimManager, QSettings* _pSettings, QWidget *parent /*= 0*/)
	: CMusenDialog(parent), m_pSettings(_pSettings)
{
	ui.setupUi(this);
	m_pSimulatorManager = _pSimManager;
	m_bSimulationStarted = false;
	m_pDEMThreadNew = nullptr;
	m_pQTThread = nullptr;
	m_bShowSimDomain = false;

	int nGPUs;
	cudaGetDeviceCount(&nGPUs);
	if (nGPUs == 0)
	{
		auto* model = qobject_cast<QStandardItemModel*>(ui.comboSimulatorType->model());
		model->item(1)->setFlags(model->item(1)->flags() & ~Qt::ItemIsEnabled);
		model->item(2)->setFlags(model->item(2)->flags() & ~Qt::ItemIsEnabled);
	}

	InitializeConnections();
	ui.accelerationTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.accelerationTable->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.boundingBoxTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.boundingBoxTable->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	m_sHelpFileName = "Users Guide/Simulator.pdf";

	ui.checkBoxCollisions->setVisible(false);
	ui.labelLBText->setVisible(false);
	ui.labelLB->setVisible(false);
}

void CSimulatorTab::InitializeConnections() const
{
	connect(ui.startButton,          &QPushButton::clicked, this, &CSimulatorTab::StartButtonClicked);
	connect(ui.stopButton,           &QPushButton::clicked, this, &CSimulatorTab::StopSimulation);
	connect(ui.updateRayleighButton, &QPushButton::clicked, this, &CSimulatorTab::UpdateRayleighTime);

	// simulation domain
	connect(ui.recalculateSimVolume, &QPushButton::clicked,      this, &CSimulatorTab::RecalculateSimulationDomain);
	connect(ui.boundingBoxTable,     &QTableWidget::itemChanged, this, &CSimulatorTab::SimDomainChanged);

	// signals from timer
	connect(&m_UpdateTimer, &QTimer::timeout, this, &CSimulatorTab::UpdateSimulationStatistics);

	connect(this, &CSimulatorTab::SimulatorStatusChanged, this, &CSimulatorTab::UpdateGUI);

	// signals to set new values directly to simulator
	connect(ui.simulationStep,     &QLineEdit::editingFinished,                         this, &CSimulatorTab::SetParameters);
	connect(ui.savingStep,         &QLineEdit::editingFinished,                         this, &CSimulatorTab::SetParameters);
	connect(ui.endTime,            &QLineEdit::editingFinished,                         this, &CSimulatorTab::SetParameters);
	connect(ui.accelerationTable,  &QTableWidget::itemChanged,                          this, &CSimulatorTab::SetParameters);
	connect(ui.checkBoxCollisions, &QCheckBox::stateChanged,                            this, &CSimulatorTab::SetParameters);
	connect(ui.comboSimulatorType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CSimulatorTab::SetParameters);

	// signals about selective saving
	connect(ui.groupBoxSelectiveSaving,		  &QGroupBox::clicked,   this, &CSimulatorTab::SetSelectiveSaving);
	connect(ui.pushButtonConfSelectiveSaving, &QPushButton::clicked, this, &CSimulatorTab::ConfigureSelectiveSaving);

	// signals about models
	connect(ui.pushButtonModels, &QPushButton::clicked, this, &CSimulatorTab::ConfigureModels);
}

void CSimulatorTab::UpdateWholeView()
{
	m_bAvoidSignal = true;
	CBaseSimulator* pSim = m_pSimulatorManager->GetSimulatorPtr();
	ShowConvLabel(ui.simStepLabel, "Simulation step", EUnitType::TIME);
	ShowConvLabel(ui.savingStepLabel, "Saving step", EUnitType::TIME);
	ShowConvLabel(ui.endTimeLabel, "End time", EUnitType::TIME);
	ShowVectorInTableRow(pSim->GetExternalAccel(), ui.accelerationTable, 0, 0);
	ShowConvValue(ui.simulationStep, pSim->GetInitSimulationStep(), EUnitType::TIME);
	ShowConvValue(ui.savingStep, pSim->GetSavingStep(), EUnitType::TIME);
	ShowConvValue(ui.endTime, pSim->GetEndTime(), EUnitType::TIME);
	UpdateSimulationDomain();
	UpdateCollisionsFlag();
	ui.comboSimulatorType->setCurrentIndex(static_cast<unsigned>(pSim->GetType()) - 1);
	ui.groupBoxSelectiveSaving->setChecked(pSim->IsSelectiveSavingEnabled());
	UpdateRayleighTime();
	UpdateModelsView();
	m_bAvoidSignal = false;
}

void CSimulatorTab::UpdateModelsView()
{
	const CModelManager* pManager = m_pSimulatorManager->GetSimulatorPtr()->GetModelManager();
	const CAbstractDEMModel *pPP = pManager->GetModel(EMusenModelType::PP);
	const CAbstractDEMModel *pPW = pManager->GetModel(EMusenModelType::PW);
	const CAbstractDEMModel *pSB = pManager->GetModel(EMusenModelType::SB);
	const CAbstractDEMModel *pLB = pManager->GetModel(EMusenModelType::LB);
	const CAbstractDEMModel *pEF = pManager->GetModel(EMusenModelType::EF);
	const CAbstractDEMModel *pHTPP = pManager->GetModel(EMusenModelType::PPHT);
	ui.labelPP->setText(pPP ? ss2qs(pPP->GetName()) : "-");
	ui.labelPW->setText(pPW ? ss2qs(pPW->GetName()) : "-");
	ui.labelSB->setText(pSB ? ss2qs(pSB->GetName()) : "-");
	ui.labelLB->setText(pLB ? ss2qs(pLB->GetName()) : "-");
	ui.labelEF->setText(pEF ? ss2qs(pEF->GetName()) : "-");
	ui.labelHT_PP->setText(pHTPP ? ss2qs(pHTPP->GetName()) : "-");
}

void CSimulatorTab::UpdateCollisionsFlag() const
{
	const ESimulatorType type = m_pSimulatorManager->GetSimulatorPtr()->GetType();
	if (type == ESimulatorType::CPU)
	{
		ui.checkBoxCollisions->setChecked(dynamic_cast<CCPUSimulator*>(m_pSimulatorManager->GetSimulatorPtr())->IsCollisionsAnalysisEnabled());
		ui.checkBoxCollisions->setEnabled(true);
	}
	else if (type == ESimulatorType::GPU)
	{
		ui.checkBoxCollisions->setChecked(false);
		ui.checkBoxCollisions->setEnabled(false);
	}
}

void CSimulatorTab::SetSelectiveSaving()
{
	m_pSimulatorManager->GetSimulatorPtr()->SetSelectiveSaving(ui.groupBoxSelectiveSaving->isChecked());
}

void CSimulatorTab::ConfigureSelectiveSaving()
{
	CSelectiveSavingTab m_pSelectiveSavingTab(m_pSimulatorManager);
	m_pSelectiveSavingTab.exec();
}

void CSimulatorTab::ConfigureModels()
{
	CModelsConfiguratorTab tab(m_pSimulatorManager->GetSimulatorPtr()->GetModelManager(), m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() == ERunningStatus::PAUSED, this);
	tab.exec();
	UpdateModelsView();
}

void CSimulatorTab::UpdateRayleighTime()
{
	ui.RecommendedTimeStep->setText(QString::number(m_pUnitConverter->GetValue(EUnitType::TIME, m_pSystemStructure->GetRecommendedTimeStep())) + " [" + ss2qs(m_pUnitConverter->GetSelectedUnit(EUnitType::TIME)) + "]");
}

void CSimulatorTab::StartButtonClicked()
{
	if (!m_bSimulationStarted)	// start new simulation
		StartSimulation();
	else if (m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() == ERunningStatus::PAUSED)	// resume simulation
		ResumeSimulation();
	else
		PauseSimulation();
}

void CSimulatorTab::StartSimulation()
{
	if (m_bSimulationStarted) return;

	SetParameters();

	// check selective saving
	if (ui.groupBoxSelectiveSaving->isChecked())
	{
		if (QMessageBox::question(this, "Confirmation", "Selective saving mode is enabled. Some properties of objects will not be saved during simulation. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
			return;
	}

	// check that scene has been saved
	if (m_pSystemStructure->GetFileName().empty())
	{
		ui.statusMessage->setText("Error: Scene should be saved before simulation");
		return;
	}

	//check general data
	std::string sErrorMessage = m_pSimulatorManager->GetSimulatorPtr()->IsDataCorrect();
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}

	// check materials database
	sErrorMessage = m_pSystemStructure->m_MaterialDatabase.IsDataCorrect();
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(QString::fromStdString(sErrorMessage));
		return;
	}

	// check that all object are in simulation domain
	const SVolumeType vSimDomain = m_pSystemStructure->GetSimulationDomain();
	if (!IsPointInDomain(vSimDomain, m_pSystemStructure->GetMaxCoordinate(0)) || !IsPointInDomain(vSimDomain, m_pSystemStructure->GetMinCoordinate(0)))
		if (QMessageBox::question(this, "Confirmation", "Some objects are not in simulation domain. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
			return;

	// check contact radius
	if (!m_pSystemStructure->IsContactRadiusEnabled())
	{
		bool bWrongContactRadius = false;
		for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
		{
			auto* particle = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(i));
			if (!particle) continue;
			if (particle->GetRadius() != particle->GetContactRadius())
			{
				bWrongContactRadius = true;
				break;
			}
		}
		if (bWrongContactRadius)
			if (QMessageBox::question(this, "Confirmation", "The contact radius of some particles differs from their real radius. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
				return;
	}

	const CGenerationManager* pGenerationManager = m_pSimulatorManager->GetSimulatorPtr()->GetGenerationManager();
	const CModelManager* pModelManager = m_pSimulatorManager->GetSimulatorPtr()->GetModelManager();

	// check that all dynamic generators are situated in simulation domain
	bool bAllGeneratorsInDomain = true;
	for (unsigned i = 0; i < pGenerationManager->GetGeneratorsNumber(); i++)
	{
		const CObjectsGenerator* pGenerator = pGenerationManager->GetGenerator(i);
		if (!pGenerator->m_bActive) continue;
		CAnalysisVolume* pVolume = m_pSystemStructure->AnalysisVolume(pGenerator->m_sVolumeKey);
		if (pVolume == nullptr) return;
		const SVolumeType vGenerationDomain = pVolume->BoundingBox(0);
		if (!IsPointInDomain(vSimDomain, vGenerationDomain.coordBeg) || !IsPointInDomain(vSimDomain, vGenerationDomain.coordEnd))
		{
			bAllGeneratorsInDomain = false;
			break;
		}
	}
	if (!bAllGeneratorsInDomain)
		if (QMessageBox::question(this, "Confirmation", "Some dynamic generators are not in simulation domain. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
			return;

	// check that there already some time points
	if (m_pSettings->value("ASK_TD_DATA_REMOVAL").toBool())
		if (m_pSystemStructure->GetAllTimePoints().size() > 1)
			if (QMessageBox::question(this, "Confirmation", "Time-dependent data in scene will be overwritten. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
				return;

	// check contact models
	sErrorMessage = pModelManager->GetModelError(EMusenModelType::PP);
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}
	sErrorMessage = pModelManager->GetModelError(EMusenModelType::PW);
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}
	sErrorMessage = pModelManager->GetModelError(EMusenModelType::SB);
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}
	sErrorMessage = pModelManager->GetModelError(EMusenModelType::LB);
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}
	sErrorMessage = pModelManager->GetModelError(EMusenModelType::EF);
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}
	sErrorMessage = pModelManager->GetModelError(EMusenModelType::PPHT);
	if (!sErrorMessage.empty())
	{
		ui.statusMessage->setText(ss2qs(sErrorMessage));
		return;
	}

	const auto simType = static_cast<ESimulatorType>(ui.comboSimulatorType->currentIndex() + 1);

	// check that all necessary models are defined
	if (m_pSystemStructure->GetNumberOfSpecificObjects(SPHERE) != 0 || (pGenerationManager->GetActiveGeneratorsNumber() != 0))
	{
		if (!pModelManager->IsModelDefined(EMusenModelType::PP))
		{
			if (QMessageBox::question(this, "Confirmation", "Particle-particle contact model is not specified. Particle-particle contacts will not be considered during the simulation. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
				return;
		}
		else if (simType == ESimulatorType::GPU && !pModelManager->IsModelGPUCompatible(EMusenModelType::PP))
		{
			ui.statusMessage->setText(ss2qs("Selected particle-particle model has no GPU support"));
			return;
		}
	}
	if (m_pSystemStructure->GeometriesNumber() != 0)
	{
		if (!pModelManager->IsModelDefined(EMusenModelType::PW))
		{
			if (QMessageBox::question(this, "Confirmation", "Particle-wall contact model is not specified. Particle-wall contacts will not be considered during the simulation. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
				return;
		}
		else if ((simType == ESimulatorType::GPU) && !pModelManager->IsModelGPUCompatible(EMusenModelType::PW))
		{
			ui.statusMessage->setText(ss2qs("Selected particle-wall model has no GPU support"));
			return;
		}
	}
	if (m_pSystemStructure->GetNumberOfSpecificObjects(SOLID_BOND) != 0 || pGenerationManager->GetActiveGeneratorsNumber() != 0)
	{
		if (!pModelManager->IsModelDefined(EMusenModelType::SB))
		{
			if (QMessageBox::question(this, "Confirmation", "Solid bond model is not specified. Solid bonds will not be considered during the simulation. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
				return;
		}
		else if (simType == ESimulatorType::GPU && !pModelManager->IsModelGPUCompatible(EMusenModelType::SB))
		{
			ui.statusMessage->setText(ss2qs("Selected solid bond model has no GPU support"));
			return;
		}
	}

	// check whether all particles are within the PBC boundaries
	SPBC pbc = m_pSystemStructure->GetPBC();
	if (pbc.bEnabled)
	{
		std::vector<CPhysicalObject*> vSpheres = m_pSystemStructure->GetAllActiveObjects(0, SPHERE);
		for (size_t i = 0; i < vSpheres.size(); ++i)
			if (!pbc.IsCoordInPBC(vSpheres[i]->GetCoordinates(0), 0))
			{
				ui.statusMessage->setText(ss2qs("Some objects are not in the domain of PBC: Object ID:") + QString::number(vSpheres[i]->m_lObjectID));
				return;
			}
	}

	// check that during PBC collision saving is turned off
	if (pbc.bEnabled)
		if (ui.checkBoxCollisions->isChecked())
		{
			ui.statusMessage->setText(ss2qs("Collision saving cannot be used with periodic boundaries"));
			return;
		}

	// check motion of geometries is properly configured
	for (const auto& g : m_pSystemStructure->AllGeometries())
		if (!g->Motion()->IsValid())
		{
			ui.statusMessage->setText("Error in geometry " + QString::fromStdString(g->Name()) + ". " + QString::fromStdString(g->Motion()->ErrorMessage()));
			return;
		}
	for (const auto& v : m_pSystemStructure->AllAnalysisVolumes())
		if (!v->Motion()->IsValid())
		{
			ui.statusMessage->setText("Error in volume " + QString::fromStdString(v->Name()) + ". " + QString::fromStdString(v->Motion()->ErrorMessage()));
			return;
		}

	m_pDEMThreadNew = new CSimulatorThread();
	m_pDEMThreadNew->m_pSimulator = m_pSimulatorManager->GetSimulatorPtr();

	emit DisableOpenGLView();
	emit SimulatorStatusChanged(ERunningStatus::RUNNING);

	m_pQTThread = new QThread();
	connect(m_pQTThread, SIGNAL(started()), m_pDEMThreadNew, SLOT(StartSimulation()));
	connect(m_pDEMThreadNew, SIGNAL(finished()), this, SLOT(SimulationFinished()));
	m_pDEMThreadNew->moveToThread(m_pQTThread);
	m_pQTThread->start();

	// start update timer
	m_UpdateTimer.start(100);
	m_bSimulationStarted = true;
}

void CSimulatorTab::ResumeSimulation()
{
	if (!m_bSimulationStarted) return;
	SetParameters();

	emit DisableOpenGLView();
	emit SimulatorStatusChanged(ERunningStatus::RUNNING);

	m_pQTThread->start();

	// start update timer
	m_UpdateTimer.start(100);
}

void CSimulatorTab::StopSimulation()
{
	if (!m_bSimulationStarted) return;

	ui.stopButton->setEnabled(false);
	//! disable old timer to avoid updating from NULL class
	m_UpdateTimer.stop();
	m_pDEMThreadNew->StopSimulation();

	while (m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() != ERunningStatus::IDLE)
		m_pQTThread->wait(100);

	SimulationFinished();
}

void CSimulatorTab::PauseSimulation()
{
	if (m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() != ERunningStatus::RUNNING) return;

	ui.stopButton->setEnabled(false);
	//! disable old timer to avoid updating from NULL class
	m_UpdateTimer.stop();
	m_pDEMThreadNew->PauseSimulation();

	while (m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() != ERunningStatus::PAUSED)
		m_pQTThread->wait(100);

	m_pQTThread->exit();

	UpdateSimulationStatistics();

	emit EnableOpenGLView();
	emit SimulatorStatusChanged(ERunningStatus::PAUSED);
	emit NumberOfTimePointsChanged();
}

void CSimulatorTab::SimulationFinished()
{
	if (m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() == ERunningStatus::PAUSED) return;

	m_UpdateTimer.stop();
	if (m_pQTThread != nullptr)
	{
		m_pQTThread->exit();
		m_pQTThread = nullptr;
		delete m_pDEMThreadNew;
		m_pDEMThreadNew = nullptr;
	}
	UpdateSimulationStatistics();
	m_bSimulationStarted = false;
	emit EnableOpenGLView();
	emit SimulatorStatusChanged(ERunningStatus::IDLE);
	emit NumberOfTimePointsChanged();
}

void CSimulatorTab::UpdateSimulationStatistics() const
{
	const auto& sim = m_pSimulatorManager->GetSimulatorPtr();

	ui.statTable->item(EStatTable::SIM_TIME, 0)->setText(QString::number(sim->GetCurrentTime()));
	ui.statTable->item(EStatTable::SIM_TIME_STEP, 0)->setText(QString::number(sim->GetCurrSimulationStep()));
	ui.statTable->item(EStatTable::MAX_PART_VELO, 0)->setText(QString::number(sim->GetMaxParticleVelocity()));
	ui.statTable->item(EStatTable::NUM_BROKEN_S_BONDS, 0)->setText(QString::number(sim->GetNumberOfBrokenBonds()));
	ui.statTable->item(EStatTable::NUM_BROKEN_L_BONDS, 0)->setText(QString::number(sim->GetNumberOfBrokenLiquidBonds()));
	ui.statTable->item(EStatTable::NUM_GENERATED, 0)->setText(QString::number(sim->GetNumberOfGeneratedObjects()));
	ui.statTable->item(EStatTable::NUM_INACTIVE, 0)->setText(QString::number(sim->GetNumberOfInactiveParticles()));

	const auto now = std::chrono::system_clock::now();
	const auto startTimePointQt = QDateTime::fromSecsSinceEpoch(std::chrono::system_clock::to_time_t(sim->GetStartDateTime()));
	const auto finishTimePoint = sim->GetFinishDateTime();
	const auto finishTimePointQt = QDateTime::fromSecsSinceEpoch(std::chrono::system_clock::to_time_t(finishTimePoint));
	const auto remainingMs = std::chrono::duration_cast<std::chrono::milliseconds>(finishTimePoint - now).count();
	const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - sim->GetStartDateTime()).count();

	ui.statTable->item(EStatTable::SIM_STARTED, 0)->setText(startTimePointQt.toString("dd.MM hh:mm:ss"));
	ui.statTable->item(EStatTable::SIM_FINISHED, 0)->setText(finishTimePointQt.toString("dd.MM hh:mm:ss"));
	ui.statTable->item(EStatTable::SIM_LEFT, 0)->setText(QString::fromStdString(MsToTimeSpan(remainingMs >= 0 ? remainingMs : 0)));
	ui.statTable->item(EStatTable::SIM_ELAPSED, 0)->setText(QString::fromStdString(MsToTimeSpan(elapsedMs >= 0 ? elapsedMs : 0)));
}

void CSimulatorTab::SetParameters()
{
	if (m_bAvoidSignal) return;
	m_bAvoidSignal = true;
	m_pSimulatorManager->GetSimulatorPtr()->SetExternalAccel(GetVectorFromTableRow(ui.accelerationTable, 0, 0));
	m_pSimulatorManager->GetSimulatorPtr()->SetInitSimulationStep(GetConvValue(ui.simulationStep, EUnitType::TIME));
	if (m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() == ERunningStatus::PAUSED)
		m_pSimulatorManager->GetSimulatorPtr()->SetCurrSimulationStep(GetConvValue(ui.simulationStep, EUnitType::TIME));
	m_pSimulatorManager->GetSimulatorPtr()->SetSavingStep(GetConvValue(ui.savingStep, EUnitType::TIME));
	m_pSimulatorManager->GetSimulatorPtr()->SetEndTime(GetConvValue(ui.endTime, EUnitType::TIME));
	m_pSimulatorManager->SetSimulatorType(static_cast<ESimulatorType>(ui.comboSimulatorType->currentIndex() + 1));

	UpdateCollisionsFlag();

	m_bAvoidSignal = false;
}

void CSimulatorTab::SimDomainChanged()
{
	SVolumeType simDomain;
	simDomain.coordBeg = GetVectorFromTableColumn( ui.boundingBoxTable, 0, 0, EUnitType::LENGTH );
	simDomain.coordEnd = GetVectorFromTableColumn( ui.boundingBoxTable, 0, 1, EUnitType::LENGTH );
	m_pSystemStructure->SetSimulationDomain(simDomain);
	emit UpdateOpenGLView();
}

void CSimulatorTab::RecalculateSimulationDomain()
{
	SVolumeType vSimDomain;
	vSimDomain.coordBeg = m_pSystemStructure->GetMinCoordinate( 0 );
	vSimDomain.coordEnd = m_pSystemStructure->GetMaxCoordinate( 0 );

	// extend simulation domain with dynamic generators
	for ( unsigned i = 0; i<m_pSimulatorManager->GetSimulatorPtr()->GetGenerationManager()->GetGeneratorsNumber(); i++ )
	{
		const CObjectsGenerator* pGenerator = m_pSimulatorManager->GetSimulatorPtr()->GetGenerationManager()->GetGenerator( i );
		if ( !pGenerator->m_bActive ) continue;
		CAnalysisVolume* pVolume = m_pSystemStructure->AnalysisVolume( pGenerator->m_sVolumeKey );
		if ( pVolume == nullptr ) continue;
		const SVolumeType vGenerationDomain = pVolume->BoundingBox(0);
		vSimDomain.coordBeg = Min(vSimDomain.coordBeg, vGenerationDomain.coordBeg);
		vSimDomain.coordEnd = Max(vSimDomain.coordEnd, vGenerationDomain.coordEnd);
	}
	vSimDomain.coordBeg = vSimDomain.coordBeg - 0.1*(vSimDomain.coordEnd - vSimDomain.coordBeg); // 10%
	vSimDomain.coordEnd = vSimDomain.coordEnd + 0.1*(vSimDomain.coordEnd - vSimDomain.coordBeg); // 10%
	m_pSystemStructure->SetSimulationDomain( vSimDomain );
	UpdateSimulationDomain();
	SimDomainChanged();

	emit UpdateOpenGLView();
}

void CSimulatorTab::UpdateSimulationDomain()
{
	QSignalBlocker blocker(ui.boundingBoxTable);

	SVolumeType vSimVolume = m_pSystemStructure->GetSimulationDomain();
	ShowVectorInTableColumn( vSimVolume.coordBeg, ui.boundingBoxTable, 0, 0, EUnitType::LENGTH );
	ShowVectorInTableColumn( vSimVolume.coordEnd, ui.boundingBoxTable, 0, 1, EUnitType::LENGTH );
	// set new units
	ShowConvLabel( ui.boundingBoxTable->horizontalHeaderItem( 0 ), "", EUnitType::LENGTH );
	ShowConvLabel( ui.boundingBoxTable->horizontalHeaderItem( 1 ), "", EUnitType::LENGTH );
}

void CSimulatorTab::UpdateGUI(ERunningStatus _status)
{
	switch(_status)
	{
	case ERunningStatus::IDLE:
		ui.statusMessage->setText("Simulation finished");
		ui.startButton->setIcon(QIcon(":/QT_GUI/Pictures/play.png"));
		ui.stopButton->setEnabled(false);
		ui.groupBoxSimOption->setEnabled(true);
		ui.updateRayleighButton->setEnabled(true);
		ui.groupBoxExternAccel->setEnabled(true);
		ui.groupBoxSimDomain->setEnabled(true);
		ui.checkBoxCollisions->setEnabled(true);
		ui.pushButtonModels->setEnabled(true);
		ui.comboSimulatorType->setEnabled(true);
		ui.groupBoxSelectiveSaving->setEnabled(true);
		break;
	case ERunningStatus::RUNNING:
		ui.statusMessage->setText("Simulation in progress...");
		ui.stopButton->setEnabled(true);
		ui.startButton->setIcon(QIcon(":/QT_GUI/Pictures/pause.png"));
		ui.groupBoxSimOption->setEnabled(false);
		ui.updateRayleighButton->setEnabled(false);
		ui.groupBoxExternAccel->setEnabled(false);
		ui.groupBoxSimDomain->setEnabled(false);
		ui.checkBoxCollisions->setEnabled(false);
		ui.pushButtonModels->setEnabled(false);
		ui.comboSimulatorType->setEnabled(false);
		ui.groupBoxSelectiveSaving->setEnabled(false);
		break;
	case ERunningStatus::PAUSED:
		ui.statusMessage->setText("Simulation paused");
		ui.startButton->setIcon(QIcon(":/QT_GUI/Pictures/play.png"));
		ui.stopButton->setEnabled(false);
		ui.groupBoxSimOption->setEnabled(true);
		ui.updateRayleighButton->setEnabled(true);
		ui.groupBoxExternAccel->setEnabled(true);
		ui.groupBoxSimDomain->setEnabled(false);
		ui.checkBoxCollisions->setEnabled(false);
		ui.pushButtonModels->setEnabled(true);
		ui.comboSimulatorType->setEnabled(false);
		ui.groupBoxSelectiveSaving->setEnabled(false);
		break;
	default:
		break;
	}
}
