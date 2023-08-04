/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ExportAsTextTab.h"
#include "qtOperations.h"
#include <QFileDialog>
#include <QThread>
#include <QMessageBox>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Worker

CExportWorker::CExportWorker(CExportAsText* _exporter, QObject* _parent)
	: QObject{ _parent }
	, m_exporter{ _exporter }
{}

void CExportWorker::StartExporting()
{
	m_exporter->Export();
	emit finished();
}

void CExportWorker::StopExporting() const
{
	if (m_exporter->GetStatus() == ERunningStatus::RUNNING)
		m_exporter->RequestStop();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Tab

CExportAsTextTab::CExportAsTextTab(CPackageGenerator* _pakageGenerator, CBondsGenerator* _bondsGenerator, QWidget* _parent)
	: CMusenDialog{ _parent }
	, m_packageGenerator{ _pakageGenerator }
	, m_bondsGenerator{ _bondsGenerator }
{
	ui.setupUi(this);

	// setup constraints widget
	ui.constraintsTab->SetGeometriesVisible(false);
	ui.constraintsTab->SetMaterials2Visible(false);
	ui.constraintsTab->SetDiameters2Visible(false);

	ui.tabWidget->setEnabled(false);

	// regular expression for positive floating point numbers
	const QRegExp regExpPosFloat("^[0-9]*[.]?[0-9]+(?:[eE][-+]?[0-9]+)?$");
	// set regular expression for limitation of input in QLineEdits
	ui.lineTimeStep->setValidator(new QRegExpValidator(regExpPosFloat, this));
	ui.lineTimeBeg ->setValidator(new QRegExpValidator(regExpPosFloat, this));
	ui.lineTimeEnd ->setValidator(new QRegExpValidator(regExpPosFloat, this));

	InitializeConnections();
	m_sHelpFileName = "Users Guide/Export as text.pdf";
}

void CExportAsTextTab::SetPointers(CSystemStructure* _systemStructure, CUnitConvertor* _unitConvertor, CMaterialsDatabase* _materialsDB, CGeometriesDatabase* _geometriesDB, CAgglomeratesDatabase* _agglomeratesDB)
{
	CMusenDialog::SetPointers(_systemStructure, _unitConvertor, _materialsDB, _geometriesDB, _agglomeratesDB);
	m_constraints.SetPointers(_systemStructure, _materialsDB);
	ui.constraintsTab->SetPointers(_systemStructure, _unitConvertor, _materialsDB, _geometriesDB, _agglomeratesDB);
	ui.constraintsTab->SetConstraintsPtr(&m_constraints);
	m_exporter.SetPointers(_systemStructure, &m_constraints, m_packageGenerator, m_bondsGenerator);
}

void CExportAsTextTab::setVisible(bool _visible)
{
	CMusenDialog::setVisible(_visible);
	if (_visible)
		UpdateWholeView();
}

void CExportAsTextTab::UpdateWholeView()
{
	if (!isVisible()) return;
	UpdatePrecision();        // update precision value
	UpdateAllFlags(); // update view and time-related values
	UpdateOrientationFlag();  // update state of Orientation check box
}

void CExportAsTextTab::InitializeConnections() const
{
	// radioButtons
	connect(ui.radioSaveAll      , &QRadioButton::clicked, this, &CExportAsTextTab::UpdateAllFlags);
	connect(ui.radioSaveSelective, &QRadioButton::clicked, this, &CExportAsTextTab::UpdateAllFlags);

	// buttons
	connect(ui.buttonExport    , &QPushButton::clicked, this, &CExportAsTextTab::ExportPressed);
	connect(ui.buttonCancel    , &QPushButton::clicked, this, &CExportAsTextTab::reject);
	connect(ui.buttonTimeUpdate, &QPushButton::clicked, this, &CExportAsTextTab::UpdateTimeFromSimulation);

	// checkboxes
	connect(ui.groupObjectType  , &QGroupBox::toggled     , this, &CExportAsTextTab::SetEnabledObjectWidgets);
	connect(ui.checkTypeParts   , &QCheckBox::stateChanged, this, &CExportAsTextTab::SetEnabledTDWidgets);
	connect(ui.checkTypeBonds   , &QCheckBox::stateChanged, this, &CExportAsTextTab::SetEnabledTDWidgets);
	connect(ui.checkTypeWalls   , &QCheckBox::stateChanged, this, &CExportAsTextTab::SetEnabledTDWidgets);

	// time data
	connect(ui.lineTimeBeg   , &QLineEdit::editingFinished, this, &CExportAsTextTab::UpdateTime);
	connect(ui.lineTimeEnd   , &QLineEdit::editingFinished, this, &CExportAsTextTab::UpdateTime);
	connect(ui.lineTimeStep  , &QLineEdit::editingFinished, this, &CExportAsTextTab::UpdateTime);
	connect(ui.radioTimeSaved, &QRadioButton::clicked     , this, &CExportAsTextTab::UpdateTime);
	connect(ui.radioTimeStep , &QRadioButton::clicked     , this, &CExportAsTextTab::UpdateTime);

	// timer
	connect(&m_updateTimer, &QTimer::timeout, this, &CExportAsTextTab::UpdateProgressInfo);
}

void CExportAsTextTab::UpdateAllFlags()
{
	ui.tabWidget->setCurrentIndex(0);
	ui.tabWidget->setEnabled(!ui.radioSaveAll->isChecked());
	UpdateTimeFromSimulation();
	UpdateOrientationFlag();
}

void CExportAsTextTab::UpdateOrientationFlag() const
{
	const bool anisotropy = m_pSystemStructure->IsAnisotropyEnabled();
	ui.checkTDPartOrient->setEnabled(anisotropy);
	if (!anisotropy)
		ui.checkTDPartOrient->setChecked(false);
}

void CExportAsTextTab::UpdatePrecision() const
{
	ui.spinAddTDPrecision->setValue(m_exporter.GetPrecision());
}

void CExportAsTextTab::UpdateTime()
{
	const auto timePoints = CalculateTimePoints();
	ui.lineTimePoints->setText(QString::number(timePoints.size()));
	ui.lineTimeStep->setEnabled(!ui.radioTimeSaved->isChecked());
}

void CExportAsTextTab::UpdateTimeFromSimulation()
{
	ui.lineTimeBeg->setText(QString::number(m_pSystemStructure->GetMinTime()));
	ui.lineTimeEnd->setText(QString::number(m_pSystemStructure->GetMaxTime()));
	const auto allTP = m_pSystemStructure->GetAllTimePoints();
	ui.lineTimeStep->setText(QString::number(allTP.size() < 2 ? 1 : allTP[1] - allTP[0]));
	UpdateTime();
}

void CExportAsTextTab::UpdateProgressInfo() const
{
	ui.progressBar->setValue(static_cast<int>(m_exporter.GetProgress()));
	ui.labelStatus->setText(QString::fromStdString(m_exporter.GetStatusMessage()));
}

void CExportAsTextTab::SetEnabledObjectWidgets(bool _active) const
{
	ui.groupConst->setEnabled(_active);
	ui.groupTDParts->setEnabled(_active);
	ui.groupTDBonds->setEnabled(_active);
	ui.groupTDWalls->setEnabled(_active);
	ui.groupTime->setEnabled(_active);
}

void CExportAsTextTab::SetEnabledTDWidgets() const
{
	ui.groupTDParts->setEnabled(ui.checkTypeParts->isChecked());
	ui.groupTDBonds->setEnabled(ui.checkTypeBonds->isChecked());
	ui.groupTDWalls->setEnabled(ui.checkTypeWalls->isChecked());
}

void CExportAsTextTab::SetEnabledAll(bool _enabled) const
{
	if (ui.radioSaveSelective->isChecked())
		ui.tabWidget->setEnabled(_enabled);
	ui.radioSaveSelective->setEnabled(_enabled);
	ui.radioSaveAll->setEnabled(_enabled);
}

std::vector<double> CExportAsTextTab::CalculateTimePoints()
{
	std::vector<double> res;

	if (ui.radioSaveAll->isChecked())
		res = m_pSystemStructure->GetAllTimePoints();
	else
	{
		const double timeMin = ui.lineTimeBeg->text().toDouble();
		const double timeMax = ui.lineTimeEnd->text().toDouble();
		if (timeMin > timeMax)
		{
			QMessageBox::critical(this, "Wrong parameters", "Time 'From' is larger than Time 'To'.");
			return {};
		}
		if (ui.radioTimeSaved->isChecked())
		{
			auto allTP = m_pSystemStructure->GetAllTimePoints();
			const double tolerance = allTP.size() > 1 ? (allTP.back() - allTP.front()) * 1e-6 : 0.0;
			std::copy_if(allTP.begin(), allTP.end(), std::back_inserter(res), [&](double t)
			{
				return t >= timeMin - tolerance && t <= timeMax + tolerance;
			});
		}
		else
		{
			const double timeStep = ui.lineTimeStep->text().toDouble();
			const auto num = static_cast<size_t>((timeMax - timeMin) / timeStep);
			for (size_t i = 0; i <= num; ++i)
				res.push_back(timeMin + i * timeStep);
		}
	}

	if (res.empty())
		res.push_back(0);

	return res;
}

void CExportAsTextTab::ApplyAllFlags()
{
	constexpr auto SetFlags = [](SBaseFlags* _flags, const QGroupBox* _group, const std::initializer_list<QCheckBox*>& _boxes)
	{
		if (!_group->isChecked())
			_flags->SetAll(false);
		else
		{
			std::vector<bool> vals;
			for (const auto* b : _boxes)
				vals.push_back(b->isChecked());
			_flags->SetFlags(vals);
		}
	};

	CExportAsText::SExportSelector settings;

	if (ui.radioSaveAll->isChecked())
		settings.SetAll(true);
	else
	{
		SetFlags(&settings.objectTypes, ui.groupObjectType, { ui.checkTypeParts, ui.checkTypeBonds, ui.checkTypeWalls });
		SetFlags(&settings.constProps , ui.groupConst     , { ui.checkConstID, ui.checkConstType, ui.checkConstGeometry, ui.checkConstMaterial, ui.checkConstActivity });
		SetFlags(&settings.tdPropsPart, ui.groupTDParts   , { ui.checkTDPartAngVel, ui.checkTDPartCoord, ui.checkTDPartForce, ui.checkTDPartForceAmpl, ui.checkTDPartOrient, ui.checkTDPartPrincStress, ui.checkTDPartStressTens, ui.checkTDPartTemp, ui.checkTDPartVel });
		SetFlags(&settings.tdPropsBond, ui.groupTDBonds   , { ui.checkTDBondCoord, ui.checkTDBondForce, ui.checkTDBondForceAmpl, ui.checkTDBondTangOverl, ui.checkTDBondTemp, ui.checkTDBondTotTorque, ui.checkTDBondVel });
		SetFlags(&settings.tdPropsWall, ui.groupTDWalls   , { ui.checkTDWallCoord, ui.checkTDWallForce, ui.checkTDWallForceAmpl, ui.checkTDWallVel });
		SetFlags(&settings.sceneInfo  , ui.groupScene     , { ui.checkInfoDomain, ui.checkInfoPBC, ui.checkInfoAnisotropy, ui.checkInfoContactRadius });
		SetFlags(&settings.geometries , ui.groupGeometries, { ui.checkGeometryGeneral, ui.checkGeometryTDP, ui.checkGeometryWalls, ui.checkGeometryVolumes });
		SetFlags(&settings.materials  , ui.groupMaterials , { ui.checkMaterialCompounds, ui.checkMaterialInteractions, ui.checkMaterialMixtures });
		SetFlags(&settings.generators , ui.groupGenerators, { ui.checkGeneratorPackage, ui.checkGeneratorBonds });
	}

	m_exporter.SetSelectors(settings);
}

void CExportAsTextTab::ExportPressed()
{
	static QString fileName;

	// start button pressed
	if (m_exporter.GetStatus() == ERunningStatus::IDLE)
	{
		if (fileName.isEmpty())
			fileName = QString::fromStdString(m_pSystemStructure->GetFileName());
		const QFileInfo fi{ fileName };
		fileName = QFileDialog::getSaveFileName(this, tr("Export as text file"), fi.absolutePath(), tr("MUSEN files (*.txt);;All files (*.*);;"));
		if (fileName.simplified().isEmpty()) return;

		// update UI elements
		emit DisableOpenGLView();
		emit RunningStatusChanged(ERunningStatus::RUNNING);
		SetEnabledAll(false);
		ui.buttonExport->setText("Stop");
		ui.buttonCancel->setEnabled(false);

		// setup exporter
		ApplyAllFlags();
		m_exporter.SetFileName(fileName.toStdString());
		m_exporter.SetTimePoints(CalculateTimePoints());
		m_exporter.SetPrecision(ui.spinAddTDPrecision->value());

		// run exporting
		m_exportWorker = new CExportWorker(&m_exporter);
		m_exportThread = new QThread();
		connect(m_exportThread, &QThread::started, m_exportWorker, &CExportWorker::StartExporting);
		connect(m_exportWorker, &CExportWorker::finished, this, &CExportAsTextTab::ExportingFinished);
		m_exportWorker->moveToThread(m_exportThread);
		m_exportThread->start();
		m_updateTimer.start(100);
	}
	// stop button was pressed
	else
	{
		m_exportWorker->StopExporting();
		while (m_exporter.GetStatus() == ERunningStatus::RUNNING)
			m_exportThread->wait(100);
		m_updateTimer.stop();
		QMessageBox::warning(this, "Information", tr("Export was interrupted."));
		ExportingFinished();
	}
}

void CExportAsTextTab::ExportingFinished()
{
	m_updateTimer.stop();
	if (m_exportThread != nullptr)
	{
		m_exportThread->exit();
		m_exportThread = nullptr;
		delete m_exportWorker;
		m_exportWorker = nullptr;
	}

	if (m_exporter.GetStatus() == ERunningStatus::FAILED)
		QMessageBox::warning(this, "Error", tr("%1").arg(QString::fromStdString(m_exporter.GetStatusMessage())));
	UpdateProgressInfo();
	m_exporter.SetStatus("", ERunningStatus::IDLE);
	ui.buttonExport->setText("Export");
	ui.buttonCancel->setEnabled(true);
	emit RunningStatusChanged(ERunningStatus::IDLE);
	emit EnableOpenGLView();
	SetEnabledAll(true);
}