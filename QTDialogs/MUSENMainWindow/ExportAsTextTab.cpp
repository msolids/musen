/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ExportAsTextTab.h"
#include "qtOperations.h"
#include <QFileDialog>
#include <QThread>
#include <QMessageBox>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Thread

CExportAsTextThread::CExportAsTextThread(CExportAsText* _exporter, QObject* parent) :
	QObject(parent),
	m_pExporter{ _exporter }
{}

void CExportAsTextThread::StartExporting()
{
	m_pExporter->Export();
	emit finished();
}

void CExportAsTextThread::StopExporting() const
{
	if (m_pExporter->GetCurrentStatus() != ERunningStatus::IDLE)
		m_pExporter->SetCurrentStatus(ERunningStatus::TO_BE_STOPPED);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Tab

CExportAsTextTab::CExportAsTextTab(QWidget* parent) :
	CMusenDialog(parent),
	m_pQTThread{ nullptr },
	m_pExportThread{ nullptr }
{
	ui.setupUi(this);

	ui.constraintsTab->SetGeometriesVisible(false);
	ui.constraintsTab->SetMaterials2Visible(false);
	ui.constraintsTab->SetDiameters2Visible(false);

	ui.tabWidget->setEnabled(false);

	// Regular expression for floating point numbers
	const QRegExp regExpFloat("^[0-9]*[.]?[0-9]+(?:[eE][-+]?[0-9]+)?$");
	// Set regular expression for limitation of input in QLineEdits
	ui.lineEditTimeStep->setValidator(new QRegExpValidator(regExpFloat, this));
	ui.lineEditTimeFrom->setValidator(new QRegExpValidator(regExpFloat, this));
	ui.lineEditTimeTo->setValidator(new QRegExpValidator(regExpFloat, this));

	UpdatePrecision();

	InitializeConnections();
	m_sHelpFileName = "Users Guide/Export as text.pdf";
}

void CExportAsTextTab::SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB)
{
	CMusenDialog::SetPointers(_pSystemStructure, _pUnitConvertor, _pMaterialsDB, _pGeometriesDB, _pAgglomDB);
	m_constraints.SetPointers(_pSystemStructure, _pMaterialsDB);
	ui.constraintsTab->SetPointers(_pSystemStructure, _pUnitConvertor, _pMaterialsDB, _pGeometriesDB, _pAgglomDB);
	ui.constraintsTab->SetConstraintsPtr(&m_constraints);
	m_exporter.SetPointers(_pSystemStructure, &m_constraints);
}

void CExportAsTextTab::UpdateWholeView()
{
	if (!isVisible()) return;
	UpdatePrecision();        // update precision value
	SelectiveSavingToggled(); // update view and time-related values
}

void CExportAsTextTab::InitializeConnections() const
{
	// radioButtons
	connect(ui.radioButtonSaveAll,		 &QRadioButton::clicked, this, &CExportAsTextTab::SelectiveSavingToggled);
	connect(ui.radioButtonSaveSelective, &QRadioButton::clicked, this, &CExportAsTextTab::SelectiveSavingToggled);

	// buttons
	connect(ui.pushButtonExport,	   &QPushButton::clicked, this, &CExportAsTextTab::ExportPressed);
	connect(ui.pushButtonCancel,  	   &QPushButton::clicked, this, &CExportAsTextTab::reject);
	connect(ui.toolButtonUpdateTimeTo, &QPushButton::clicked, this, &CExportAsTextTab::UpdateTimeFromSimulation);

	// checkboxes
	connect(ui.groupBoxObjectType,	   &QGroupBox::toggled,		 this, &CExportAsTextTab::SetObjectTypeCheckBoxes);
	connect(ui.checkBoxParticle,	   &QCheckBox::stateChanged, this, &CExportAsTextTab::SetRelevantCheckBoxesParticle);
	connect(ui.checkBoxSolidBond,	   &QCheckBox::stateChanged, this, &CExportAsTextTab::SetRelevantCheckBoxesSB);
	connect(ui.checkBoxTriangularWall, &QCheckBox::stateChanged, this, &CExportAsTextTab::SetRelevantCheckBoxesTW);
	connect(ui.checkBoxQuaternion,	   &QCheckBox::stateChanged, this, &CExportAsTextTab::SetQuaternionCheckBox);

	// time data
	connect(ui.lineEditTimeFrom,	 &QLineEdit::editingFinished, this, &CExportAsTextTab::SetNewTime);
	connect(ui.lineEditTimeTo,		 &QLineEdit::editingFinished, this, &CExportAsTextTab::SetNewTime);
	connect(ui.lineEditTimeStep,	 &QLineEdit::editingFinished, this, &CExportAsTextTab::SetNewTime);
	connect(ui.radioButtonTimeSaved, &QRadioButton::clicked,	  this, &CExportAsTextTab::SetNewTime);
	connect(ui.radioButtonTimeStep,  &QRadioButton::clicked,	  this, &CExportAsTextTab::SetNewTime);

	// timer
	connect(&m_UpdateTimer, &QTimer::timeout, this, &CExportAsTextTab::UpdateProgressInfo);
}

void CExportAsTextTab::UpdatePrecision() const
{
	ui.spinBoxPrecision->setValue(m_exporter.GetPrecision());
}

void CExportAsTextTab::SelectiveSavingToggled()
{
	ui.checkBoxQuaternion->setChecked(m_pSystemStructure->IsAnisotropyEnabled());
	ui.tabWidget->setCurrentIndex(0);
	ui.tabWidget->setEnabled(!ui.radioButtonSaveAll->isChecked());
	UpdateTimeFromSimulation();
}

void CExportAsTextTab::SetObjectTypeCheckBoxes(bool _active) const
{
	ui.groupBoxConstProperties->setEnabled(_active);
	ui.groupBoxTDProperties->setEnabled(_active);
	ui.groupBoxTime->setEnabled(_active);
}

void CExportAsTextTab::SetRelevantCheckBoxesParticle() const
{
	ui.checkBoxQuaternion->setEnabled(ui.checkBoxParticle->isChecked());
	ui.checkBoxAngularVelocity->setEnabled(ui.checkBoxParticle->isChecked());
	ui.checkBoxCoordinate->setEnabled(ui.checkBoxParticle->isChecked());
	ui.checkBoxStressTensor->setEnabled(ui.checkBoxParticle->isChecked());
	ui.checkBoxTemperature->setEnabled(ui.checkBoxParticle->isChecked() || ui.checkBoxSolidBond->isChecked());
}

void CExportAsTextTab::SetRelevantCheckBoxesSB() const
{
	ui.checkBoxTangOverlap->setEnabled(ui.checkBoxSolidBond->isChecked());
	ui.checkBoxTotalTorque->setEnabled(ui.checkBoxSolidBond->isChecked());
	ui.checkBoxTemperature->setEnabled(ui.checkBoxParticle->isChecked() || ui.checkBoxSolidBond->isChecked());
}

void CExportAsTextTab::SetRelevantCheckBoxesTW() const
{
	ui.checkBoxPlanesCoordinates->setEnabled(ui.checkBoxTriangularWall->isChecked());
}

void CExportAsTextTab::SetQuaternionCheckBox()
{
	if (ui.checkBoxQuaternion->isChecked())
		if (!m_pSystemStructure->IsAnisotropyEnabled())
		{
			QMessageBox::warning(this, "Wrong parameters", "Export of Quaternions is not possible. Flag 'Consider particles anisotropy' has to be set in Scene Editor.");
			ui.checkBoxQuaternion->setCheckState(Qt::CheckState::Unchecked);
		}
}

void CExportAsTextTab::SetWholeTabEnabled(bool _enabled) const
{
	if (ui.radioButtonSaveSelective->isChecked())
		ui.tabWidget->setEnabled(_enabled);
	ui.radioButtonSaveSelective->setEnabled(_enabled);
	ui.radioButtonSaveAll->setEnabled(_enabled);
}

void CExportAsTextTab::UpdateProgressInfo() const
{
	ui.progressBarExporting->setValue(int(m_exporter.GetProgressPercent()));
	ui.statusLabel->setText(ss2qs(m_exporter.GetProgressMessage()));
}

void CExportAsTextTab::SetNewTime()
{
	CalculateTimePoints();
	UpdateTimeParameters();
}

void CExportAsTextTab::CalculateTimePoints()
{
	m_vTimePoints.clear();

	if (ui.radioButtonSaveAll->isChecked())
		m_vTimePoints = m_pSystemStructure->GetAllTimePoints();
	else
	{
		const double timeMin = ui.lineEditTimeFrom->text().toDouble();
		const double timeMax = ui.lineEditTimeTo->text().toDouble();
		if (timeMin > timeMax)
		{
			QMessageBox::critical(this, "Wrong parameters", "Time 'From' is larger than Time 'To'.");
			return;
		}
		if (ui.radioButtonTimeSaved->isChecked())
		{
			auto allTP = m_pSystemStructure->GetAllTimePoints();
			std::copy_if(allTP.begin(), allTP.end(), std::back_inserter(m_vTimePoints), [&timeMin, &timeMax](double t) { return (t >= timeMin) && (t <= timeMax); });
		}
		else
		{
			const double timeStep = ui.lineEditTimeStep->text().toDouble();
			const size_t num = size_t((timeMax - timeMin) / timeStep);
			for (size_t i = 0; i <= num; ++i)
				m_vTimePoints.push_back(timeMin + i * timeStep);
		}
	}

	if (m_vTimePoints.empty())
		m_vTimePoints.push_back(0);
}

void CExportAsTextTab::UpdateTimeParameters() const
{
	ui.lineEditTimePoints->setText(QString::number(m_vTimePoints.size()));
	ui.lineEditTimeStep->setEnabled(!ui.radioButtonTimeSaved->isChecked());
}

void CExportAsTextTab::UpdateTimeFromSimulation()
{
	ui.lineEditTimeFrom->setText(QString::number(m_pSystemStructure->GetMinTime()));
	ui.lineEditTimeTo->setText(QString::number(m_pSystemStructure->GetMaxTime()));
	const auto allTP = m_pSystemStructure->GetAllTimePoints();
	ui.lineEditTimeStep->setText(QString::number(allTP.size() < 2 ? m_pSystemStructure->GetRecommendedTimeStep() : allTP[1] - allTP[0]));
	SetNewTime();
}

void CExportAsTextTab::ApplyAllFlags()
{
	CExportAsText::SObjectTypeFlags objectTypeFlags;
	CExportAsText::SSceneInfoFlags sceneInfoFlags;
	CExportAsText::SConstPropsFlags constPropsFlags;
	CExportAsText::STDPropsFlags tdPropsFlags;
	CExportAsText::SGeometriesFlags geometriesFlags;
	CExportAsText::SMaterialsFlags materialsFlags;

	if (ui.radioButtonSaveAll->isChecked())
	{
		objectTypeFlags.SetAll(true);
		sceneInfoFlags.SetAll(true);
		constPropsFlags.SetAll(true);
		tdPropsFlags.SetAll(true);
		geometriesFlags.SetAll(true);
		materialsFlags.SetAll(true);
	}
	else
	{
		if (ui.groupBoxObjectType->isChecked())
			objectTypeFlags.SetFlags({ ui.checkBoxParticle->isChecked(), ui.checkBoxSolidBond->isChecked(), ui.checkBoxLiquidBond->isChecked(), ui.checkBoxTriangularWall->isChecked() });
		else
			objectTypeFlags.SetAll(false);

		if (ui.groupBoxSceneInfo->isChecked())
			sceneInfoFlags.SetFlags({ ui.checkBoxComputationalDomain->isChecked(), ui.checkBoxBoundaryConditions->isChecked(), ui.checkBoxAnisotropy->isChecked(), ui.checkBoxContactRadius->isChecked() });
		else
			sceneInfoFlags.SetAll(false);

		if (ui.groupBoxConstProperties->isChecked())
			constPropsFlags.SetFlags({ ui.checkBoxIdObject->isChecked(), ui.checkBoxObjectType->isChecked(), ui.checkBoxObjectGeometry->isChecked(), ui.checkBoxMaterial->isChecked(), ui.checkBoxIntervalActivity->isChecked() });
		else
			constPropsFlags.SetAll(false);

		if (ui.groupBoxTDProperties->isChecked())
			tdPropsFlags.SetFlags({ ui.checkBoxCoordinate->isChecked(), ui.checkBoxVelocity->isChecked(), ui.checkBoxAngularVelocity->isChecked(), ui.checkBoxTotalForce->isChecked(), ui.checkBoxForce->isChecked(), ui.checkBoxQuaternion->isChecked(), ui.checkBoxStressTensor->isChecked(), ui.checkBoxPlanesCoordinates->isChecked(), ui.checkBoxTotalTorque->isChecked(), ui.checkBoxTangOverlap->isChecked(), ui.checkBoxTemperature->isChecked() });
		else
			tdPropsFlags.SetAll(false);

		if (ui.groupBoxGeometries->isChecked())
			geometriesFlags.SetFlags({ ui.checkBoxGeometry->isChecked(), ui.checkBoxTDPGeometry->isChecked(), ui.checkBoxIndOfPlanesGeometry->isChecked() });
		else
			geometriesFlags.SetAll(false);

		if (ui.groupBoxMaterials->isChecked())
			materialsFlags.SetFlags({ ui.checkBoxCompounds->isChecked(), ui.checkBoxInteractions->isChecked(), ui.checkBoxMixtures->isChecked() });
		else
			materialsFlags.SetAll(false);
	}

	m_exporter.SetFlags(objectTypeFlags, sceneInfoFlags, constPropsFlags, tdPropsFlags, geometriesFlags, materialsFlags);
}

void CExportAsTextTab::ExportPressed()
{
	static QString fileName;

	if (m_exporter.GetCurrentStatus() == ERunningStatus::IDLE) // start button pressed
	{
		if (fileName.isEmpty())
			fileName = QString::fromStdString(m_pSystemStructure->GetFileName());
		const QFileInfo fi(fileName);
		fileName = QFileDialog::getSaveFileName(this, tr("Export as text file"), fi.absolutePath(), tr("MUSEN files (*.txt);;All files (*.*);;"));
		if (fileName.simplified().isEmpty()) return;
		if (!IsFileWritable(fileName))
		{
			QMessageBox::warning(this, "Writing error", "Unable to save - the selected file is not writable.");
			return;
		}

		// update running status
		m_exporter.SetCurrentStatus(ERunningStatus::RUNNING);

		// update UI elements
		emit DisableOpenGLView();
		emit RunningStatusChanged(ERunningStatus::RUNNING);
		SetWholeTabEnabled(false);
		ui.pushButtonExport->setText("Stop");
		ui.pushButtonCancel->setEnabled(false);

		// setup exporter
		ApplyAllFlags();
		m_exporter.SetTimePoints(m_vTimePoints);
		m_exporter.SetFileName(qs2ss(fileName));
		m_exporter.SetPrecision(ui.spinBoxPrecision->value());

		// run exporting
		m_pExportThread = new CExportAsTextThread(&m_exporter);
		m_pQTThread = new QThread();
		connect(m_pQTThread, &QThread::started, m_pExportThread, &CExportAsTextThread::StartExporting);
		connect(m_pExportThread, &CExportAsTextThread::finished, this, &CExportAsTextTab::ExportingFinished);
		m_pExportThread->moveToThread(m_pQTThread);
		m_pQTThread->start();
		m_UpdateTimer.start(100);
	}
	else // stop button was pressed
	{
		m_pExportThread->StopExporting();
		while (m_exporter.GetCurrentStatus() != ERunningStatus::IDLE)
			m_pQTThread->wait(100);
		m_UpdateTimer.stop();
		ui.progressBarExporting->setValue(0);
		const double dRealProgressPercent = m_exporter.GetProgressPercent() > 50 ? m_exporter.GetProgressPercent() - 50 : 0;
		QMessageBox::information(this, "Information", tr("About %1% of data has been exported.").arg(unsigned(dRealProgressPercent)));
		ExportingFinished();
	}
}

void CExportAsTextTab::ExportingFinished()
{
	m_UpdateTimer.stop();
	ui.statusLabel->setText("Export finished.");
	if (m_pQTThread != nullptr)
	{
		m_pQTThread->exit();
		m_pQTThread = nullptr;
		delete m_pExportThread;
		m_pExportThread = nullptr;
	}

	m_exporter.SetCurrentStatus(ERunningStatus::IDLE);
	if (!m_exporter.GetErrorMessage().empty())
	{
		QMessageBox::warning(this, "Error", tr("%1").arg(ss2qs(m_exporter.GetErrorMessage())));
		ui.statusLabel->setText("Export error.");
		ui.progressBarExporting->setValue(0);
	}
	else
		ui.progressBarExporting->setValue(100);
	ui.pushButtonExport->setText("Export");
	ui.pushButtonCancel->setEnabled(true);
	emit RunningStatusChanged(ERunningStatus::IDLE);
	emit EnableOpenGLView();
	SetWholeTabEnabled(true);
}