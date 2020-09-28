/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ResultsAnalyzerTab.h"
#include "qtOperations.h"
#include <QMessageBox>
#include <QFileDialog>

CAnalyzerThread::CAnalyzerThread(CResultsAnalyzer *_pAnalyzer, QObject *parent /*= 0*/) : QObject(parent)
{
	m_pAnalyzer = _pAnalyzer;
	connect(&m_Thread, SIGNAL(started()), this, SLOT(StartAnalyzing()));
}

CAnalyzerThread::~CAnalyzerThread()
{
	m_Thread.quit();
	m_Thread.wait();
}

void CAnalyzerThread::Run(const QString& _sFileName)
{
	m_sFileName = _sFileName;
	this->moveToThread(&m_Thread);
	m_Thread.start();
}

void CAnalyzerThread::Stop()
{
	m_Thread.exit();
}

void CAnalyzerThread::StopAnalyzing()
{
	m_pAnalyzer->SetCurrentStatus(CResultsAnalyzer::EStatus::ShouldBeStopped);
}

void CAnalyzerThread::StartAnalyzing()
{
	m_pAnalyzer->StartExport(m_sFileName.toStdString());
	emit Finished();
}


CResultsAnalyzerTab::CResultsAnalyzerTab(QWidget *parent)
	: CMusenDialog(parent)
{
	ui.setupUi(this);

	m_pAnalyzer = nullptr;
	m_bAvoidSignal = false;
	m_pConstraintsEditorTab = new CConstraintsEditorTab(ui.tabWidget);
	m_pAnalyzerThread = nullptr;

	m_vTypesForTypeActive = { CResultsAnalyzer::EPropertyType::BondForce, CResultsAnalyzer::EPropertyType::BondNumber, CResultsAnalyzer::EPropertyType::Coordinate,
		CResultsAnalyzer::EPropertyType::CoordinationNumber, CResultsAnalyzer::EPropertyType::Deformation, CResultsAnalyzer::EPropertyType::Diameter, CResultsAnalyzer::EPropertyType::Distance,
		CResultsAnalyzer::EPropertyType::Duration, CResultsAnalyzer::EPropertyType::Energy, CResultsAnalyzer::EPropertyType::ForceNormal, CResultsAnalyzer::EPropertyType::ForceTangential,
		CResultsAnalyzer::EPropertyType::ForceTotal, CResultsAnalyzer::EPropertyType::KineticEnergy, CResultsAnalyzer::EPropertyType::Length, CResultsAnalyzer::EPropertyType::MaxOverlap,
		CResultsAnalyzer::EPropertyType::Orientation, CResultsAnalyzer::EPropertyType::PartNumber, CResultsAnalyzer::EPropertyType::PotentialEnergy, CResultsAnalyzer::EPropertyType::ResidenceTime,
		CResultsAnalyzer::EPropertyType::Strain, CResultsAnalyzer::EPropertyType::VelocityNormal, CResultsAnalyzer::EPropertyType::VelocityRotational,
		CResultsAnalyzer::EPropertyType::VelocityTangential, CResultsAnalyzer::EPropertyType::VelocityTotal, CResultsAnalyzer::EPropertyType::Stress, CResultsAnalyzer::EPropertyType::Temperature };

	m_vTypesForDistanceActive = { CResultsAnalyzer::EPropertyType::Distance };

	m_vTypesForComponentActive = { CResultsAnalyzer::EPropertyType::Coordinate, CResultsAnalyzer::EPropertyType::Distance, CResultsAnalyzer::EPropertyType::ForceNormal,
		CResultsAnalyzer::EPropertyType::ForceTangential, CResultsAnalyzer::EPropertyType::ForceTotal, CResultsAnalyzer::EPropertyType::VelocityNormal,
		CResultsAnalyzer::EPropertyType::VelocityRotational, CResultsAnalyzer::EPropertyType::VelocityTangential, CResultsAnalyzer::EPropertyType::VelocityTotal,
		CResultsAnalyzer::EPropertyType::Orientation, CResultsAnalyzer::EPropertyType::Stress };

	m_vTypesForDistrParamsActive = { CResultsAnalyzer::EPropertyType::ResidenceTime };

	resize(minimumSizeHint());
	InitializeConnections();
}

void CResultsAnalyzerTab::Initialize()
{
	SetResultsTypeVisible(false);
	SetDistanceVisible(false);
	SetComponentVisible(false);
	SetCollisionsVisible(false);
	SetGeometryVisible(false);
	SetConstraintsVisible(false);
	SetConstraintMaterialsVisible(false);
	SetConstraintMaterials2Visible(false);
	SetConstraintVolumesVisible(false);
	SetConstraintGeometriesVisible(false);
	SetConstraintDiametersVisible(false);
	SetConstraintDiameters2Visible(false);
	InitializeAnalyzerTab();
}

CResultsAnalyzerTab::~CResultsAnalyzerTab()
{
}

void CResultsAnalyzerTab::SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB)
{
	CMusenDialog::SetPointers(_pSystemStructure, _pUnitConvertor, _pMaterialsDB, _pGeometriesDB, _pAgglomDB);
	m_pAnalyzer->SetSystemStructure(_pSystemStructure);
	m_pAnalyzer->GetConstraintsPtr()->SetPointers(_pSystemStructure, _pMaterialsDB);
	m_pConstraintsEditorTab->SetPointers(_pSystemStructure, _pUnitConvertor, _pMaterialsDB, _pGeometriesDB, _pAgglomDB);
	m_pConstraintsEditorTab->SetConstraintsPtr(m_pAnalyzer->GetConstraintsPtr());
}

void CResultsAnalyzerTab::UpdateSettings()
{
	m_pAnalyzer->UpdateSettings();
	m_pConstraintsEditorTab->UpdateSettings();
}

void CResultsAnalyzerTab::InitializeConnections() const
{
	// connect ComboBoxProperty
	connect(ui.listProperty, &QListWidget::currentRowChanged, this, &CResultsAnalyzerTab::NewPropertySelected);

	// buttons
	connect(ui.pushButtonCancel, SIGNAL(clicked()), this, SLOT(reject()));
	connect(ui.pushButtonExport, SIGNAL(clicked()), this, SLOT(ExportDataPressed()));

	// results type
	connect(ui.radioButtonDistribution, &QRadioButton::clicked,			this, &CResultsAnalyzerTab::NewResultsTypeSelected);
	connect(ui.radioButtonAverage,		&QRadioButton::clicked,			this, &CResultsAnalyzerTab::NewResultsTypeSelected);
	connect(ui.radioButtonMaximum,		&QRadioButton::clicked,			this, &CResultsAnalyzerTab::NewResultsTypeSelected);
	connect(ui.radioButtonMinimum,		&QRadioButton::clicked,			this, &CResultsAnalyzerTab::NewResultsTypeSelected);
	connect(ui.lineEditDistrMin,		&QLineEdit::editingFinished,	this, &CResultsAnalyzerTab::NewDistrParamSet);
	connect(ui.lineEditDistrMax,		&QLineEdit::editingFinished,	this, &CResultsAnalyzerTab::NewDistrParamSet);
	connect(ui.lineEditDistrNClasses,	&QLineEdit::editingFinished,	this, &CResultsAnalyzerTab::NewDistrParamSet);

	// distance
	connect(ui.radioButtonToPoint, SIGNAL(clicked(bool)), this, SLOT(NewDistanceTypeSelected(bool)));
	connect(ui.radioButtonToLine, SIGNAL(clicked(bool)), this, SLOT(NewDistanceTypeSelected(bool)));
	connect(ui.lineEditX1, &QLineEdit::editingFinished, this, &CResultsAnalyzerTab::NewDataPointsSet);
	connect(ui.lineEditY1, &QLineEdit::editingFinished, this, &CResultsAnalyzerTab::NewDataPointsSet);
	connect(ui.lineEditZ1, &QLineEdit::editingFinished, this, &CResultsAnalyzerTab::NewDataPointsSet);
	connect(ui.lineEditX2, &QLineEdit::editingFinished, this, &CResultsAnalyzerTab::NewDataPointsSet);
	connect(ui.lineEditY2, &QLineEdit::editingFinished, this, &CResultsAnalyzerTab::NewDataPointsSet);
	connect(ui.lineEditZ2, &QLineEdit::editingFinished, this, &CResultsAnalyzerTab::NewDataPointsSet);

	// components
	connect(ui.radioButtonTotal, SIGNAL(clicked(bool)), this, SLOT(NewComponentSelected(bool)));
	connect(ui.radioButtonX, SIGNAL(clicked(bool)), this, SLOT(NewComponentSelected(bool)));
	connect(ui.radioButtonY, SIGNAL(clicked(bool)), this, SLOT(NewComponentSelected(bool)));
	connect(ui.radioButtonZ, SIGNAL(clicked(bool)), this, SLOT(NewComponentSelected(bool)));

	// relation
	connect(ui.radioButtonNumberRelated, SIGNAL(clicked(bool)), this, SLOT(NewRelationSelected(bool)));
	connect(ui.radioButtonFrequencyRelated, SIGNAL(clicked(bool)), this, SLOT(NewRelationSelected(bool)));

	// collisions type
	connect(ui.radioButtonPP, SIGNAL(clicked(bool)), this, SLOT(NewCollisionsTypeSelected(bool)));
	connect(ui.radioButtonPW, SIGNAL(clicked(bool)), this, SLOT(NewCollisionsTypeSelected(bool)));

	// geometry
	connect(ui.comboBoxGeometry, SIGNAL(currentIndexChanged(int)), this, SLOT(NewGeometrySelected(int)));

	// time data
	connect(ui.lineEditTimeFrom,		&QLineEdit::editingFinished,	this, &CResultsAnalyzerTab::NewTimeSet);
	connect(ui.lineEditTimeTo,			&QLineEdit::editingFinished,	this, &CResultsAnalyzerTab::NewTimeSet);
	connect(ui.lineEditTimeStep,		&QLineEdit::editingFinished,	this, &CResultsAnalyzerTab::NewTimeSet);
	connect(ui.radioButtonTimeSaved,	&QRadioButton::clicked,			this, &CResultsAnalyzerTab::NewTimeSet);
	connect(ui.radioButtonTimeStep,		&QRadioButton::clicked,			this, &CResultsAnalyzerTab::NewTimeSet);

	// constraints dialog
	connect(m_pConstraintsEditorTab, SIGNAL(finished(int)), this, SLOT(CloseDialog(int)));

	// timers
	connect(&m_UpdateTimer, SIGNAL(timeout()), this, SLOT(UpdateExportStatistics()));
}

void CResultsAnalyzerTab::SetResultsTypeVisible(bool _bVisible)
{
	ui.groupBoxResultsType->setVisible(_bVisible);
}

void CResultsAnalyzerTab::SetDistanceVisible(bool _bVisible)
{
	ui.groupBoxDistance->setVisible(_bVisible);
}

void CResultsAnalyzerTab::SetComponentVisible(bool _bVisible)
{
	ui.frameComponent->setVisible(_bVisible);
}

void CResultsAnalyzerTab::SetCollisionsVisible(bool _bVisible)
{
	ui.frameCollisions->setVisible(_bVisible);
}

void CResultsAnalyzerTab::SetGeometryVisible(bool _bVisible)
{
	ui.frameGeometries->setVisible(_bVisible);
}

void CResultsAnalyzerTab::SetPoint2Visible(bool _bVisible)
{
	ui.labelP2->setVisible(_bVisible);
	ui.lineEditX2->setVisible(_bVisible);
	ui.lineEditY2->setVisible(_bVisible);
	ui.lineEditZ2->setVisible(_bVisible);
}

void CResultsAnalyzerTab::SetConstraintsVisible(bool _bVisible)
{
	if (!_bVisible)
		ui.tabWidget->removeTab(1);
	else
	{
		ui.tabWidget->addTab(m_pConstraintsEditorTab, "Constraints");
		m_pConstraintsEditorTab->resize(m_pConstraintsEditorTab->minimumSizeHint());
	}
}

void CResultsAnalyzerTab::SetConstraintMaterialsVisible(bool _bVisible)
{
	m_pConstraintsEditorTab->SetMaterialsVisible(_bVisible);
}

void CResultsAnalyzerTab::SetConstraintMaterials2Visible(bool _bVisible)
{
	m_pConstraintsEditorTab->SetMaterials2Visible(_bVisible);
}

void CResultsAnalyzerTab::SetConstraintVolumesVisible(bool _bVisible)
{
	m_pConstraintsEditorTab->SetVolumesVisible(_bVisible);
}

void CResultsAnalyzerTab::SetConstraintGeometriesVisible(bool _bVisible)
{
	m_pConstraintsEditorTab->SetGeometriesVisible(_bVisible);
}

void CResultsAnalyzerTab::SetConstraintDiametersVisible(bool _bVisible)
{
	m_pConstraintsEditorTab->SetDiametersVisible(_bVisible);
}

void CResultsAnalyzerTab::SetConstraintDiameters2Visible(bool _bVisible)
{
	m_pConstraintsEditorTab->SetDiameters2Visible(_bVisible);
}

void CResultsAnalyzerTab::UpdateSelectedProperty()
{
	if (ui.groupBoxProperty->isHidden()) return;
	m_bAvoidSignal = true;
	UpdateSelectedResultsType();
	UpdateResultsTypeActivity();
	UpdateDistance();
	UpdateDistanceVisibility();
	UpdateComponentActivity();
	UpdateTime();
	UpdateDistrParams();
	UpdateDistrParamsActivity();
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateSelectedResultsType()
{
	if (ui.groupBoxResultsType->isHidden()) return;
	m_bAvoidSignal = true;
	switch (m_pAnalyzer->m_nResultsType)
	{
	case CResultsAnalyzer::EResultType::Distribution:
		ui.radioButtonDistribution->setChecked(true);
		break;
	case CResultsAnalyzer::EResultType::Average:
		ui.radioButtonAverage->setChecked(true);
		break;
	case CResultsAnalyzer::EResultType::Maximum:
		ui.radioButtonMaximum->setChecked(true);
		break;
	case CResultsAnalyzer::EResultType::Minimum:
		ui.radioButtonMinimum->setChecked(true);
		break;
	default:
		break;
	}
	UpdateDistrParamsActivity();
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateSelectedDistance()
{
	if (ui.groupBoxDistance->isHidden()) return;
	m_bAvoidSignal = true;
	if (m_pAnalyzer->m_nDistance == CResultsAnalyzer::EDistanceType::ToPoint)
		ui.radioButtonToPoint->setChecked(true);
	else
		ui.radioButtonToLine->setChecked(true);
	SetPoint2Visible(m_pAnalyzer->m_nDistance == CResultsAnalyzer::EDistanceType::ToLine);
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateSelectedComponent()
{
	if (ui.frameComponent->isHidden()) return;
	m_bAvoidSignal = true;
	if (m_pAnalyzer->m_nComponent== CResultsAnalyzer::EVectorComponent::Total)
		ui.radioButtonTotal->setChecked(true);
	else if (m_pAnalyzer->m_nComponent == CResultsAnalyzer::EVectorComponent::X)
		ui.radioButtonX->setChecked(true);
	else if (m_pAnalyzer->m_nComponent == CResultsAnalyzer::EVectorComponent::Y)
		ui.radioButtonY->setChecked(true);
	else if (m_pAnalyzer->m_nComponent == CResultsAnalyzer::EVectorComponent::Z)
		ui.radioButtonZ->setChecked(true);
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateSelectedRelation()
{
	if (ui.frameCollisions->isHidden()) return;
	m_bAvoidSignal = true;
	if (m_pAnalyzer->m_nRelation== CResultsAnalyzer::ERelationType::Existing)
		ui.radioButtonNumberRelated->setChecked(true);
	else if (m_pAnalyzer->m_nRelation == CResultsAnalyzer::ERelationType::Appeared)
		ui.radioButtonFrequencyRelated->setChecked(true);
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateSelectedCollisionsType()
{
	if (ui.frameCollisions->isHidden()) return;
	m_bAvoidSignal = true;
	if (m_pAnalyzer->m_nCollisionType == CResultsAnalyzer::ECollisionType::ParticleParticle)
	{
		ui.radioButtonPP->setChecked(true);
		SetConstraintGeometriesVisible(false);
		SetConstraintMaterials2Visible(true);
		SetConstraintDiameters2Visible(true);
	}
	else if (m_pAnalyzer->m_nCollisionType == CResultsAnalyzer::ECollisionType::ParticleWall)
	{
		ui.radioButtonPW->setChecked(true);
		SetConstraintGeometriesVisible(true);
		SetConstraintMaterials2Visible(false);
		SetConstraintDiameters2Visible(false);
	}
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateSelectedGeometry()
{
	if (ui.frameGeometries->isHidden()) return;
	m_bAvoidSignal = true;
	ui.comboBoxGeometry->setCurrentIndex(static_cast<int>(m_pAnalyzer->m_nGeometryIndex));
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateDistance()
{
	if (ui.groupBoxDistance->isHidden()) return;
	m_bAvoidSignal = true;
	ui.lineEditX1->setText(QString::number(m_pAnalyzer->m_Point1.x));
	ui.lineEditY1->setText(QString::number(m_pAnalyzer->m_Point1.y));
	ui.lineEditZ1->setText(QString::number(m_pAnalyzer->m_Point1.z));
	ui.lineEditX2->setText(QString::number(m_pAnalyzer->m_Point2.x));
	ui.lineEditY2->setText(QString::number(m_pAnalyzer->m_Point2.y));
	ui.lineEditZ2->setText(QString::number(m_pAnalyzer->m_Point2.z));
	SetPoint2Visible(m_pAnalyzer->m_nDistance == CResultsAnalyzer::EDistanceType::ToLine);
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateTime()
{
	m_bAvoidSignal = true;
	ui.lineEditTimeFrom->setText(QString::number(m_pAnalyzer->m_dTimeMin));
	ui.lineEditTimeTo->setText(QString::number(m_pAnalyzer->m_dTimeMax));
	if (m_pAnalyzer->m_bOnlySavedTP)
		ui.radioButtonTimeSaved->setChecked(true);
	else
	{
		ui.radioButtonTimeStep->setChecked(true);
		ui.lineEditTimeStep->setText(QString::number(m_pAnalyzer->m_dTimeStep));
	}
	UpdateTimeParams();
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateDistrParams()
{
	m_bAvoidSignal = true;
	ui.lineEditDistrMin->setText(QString::number(m_pAnalyzer->m_dPropMin));
	ui.lineEditDistrMax->setText(QString::number(m_pAnalyzer->m_dPropMax));
	ui.lineEditDistrNClasses->setText(QString::number(m_pAnalyzer->m_nPropSteps));
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateConstraints()
{
	m_bAvoidSignal = true;
	if (ui.tabWidget->count() == 2)
		m_pConstraintsEditorTab->UpdateWholeView();
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::UpdateResultsTypeActivity()
{
	if (ui.groupBoxResultsType->isHidden()) return;
	bool bActiveAll = std::find(m_vTypesForTypeActive.begin(), m_vTypesForTypeActive.end(), m_pAnalyzer->GetProperty()) != m_vTypesForTypeActive.end();
	bool bActiveParam = std::find(m_vTypesForDistrParamsActive.begin(), m_vTypesForDistrParamsActive.end(), m_pAnalyzer->GetProperty()) != m_vTypesForDistrParamsActive.end();

	ui.groupBoxResultsType->setEnabled(bActiveParam || bActiveAll);
	ui.radioButtonAverage->setEnabled(!bActiveParam);
	ui.radioButtonMaximum->setEnabled(!bActiveParam);
	ui.radioButtonMinimum->setEnabled(!bActiveParam);
	if (bActiveParam)
		ui.radioButtonDistribution->setChecked(true);
	else
		UpdateSelectedResultsType();
}

void CResultsAnalyzerTab::UpdateDistanceVisibility()
{
	bool bVisible = std::find(m_vTypesForDistanceActive.begin(), m_vTypesForDistanceActive.end(), m_pAnalyzer->GetProperty()) != m_vTypesForDistanceActive.end();
	ui.groupBoxDistance->setVisible(bVisible);
}

void CResultsAnalyzerTab::UpdateComponentActivity()
{
	if (ui.frameComponent->isHidden()) return;
	CResultsAnalyzer::EPropertyType p = m_pAnalyzer->GetProperty();
	bool bActive = std::find(m_vTypesForComponentActive.begin(), m_vTypesForComponentActive.end(), p) != m_vTypesForComponentActive.end();
	bool bNotActive = (ui.groupBoxDistance->isVisible() &&
		p == CResultsAnalyzer::EPropertyType::Distance && m_pAnalyzer->m_nDistance == CResultsAnalyzer::EDistanceType::ToLine);
	ui.frameComponent->setEnabled(bActive && !bNotActive);
}

void CResultsAnalyzerTab::UpdateDistrParamsActivity()
{
	CResultsAnalyzer::EPropertyType p = m_pAnalyzer->GetProperty();
	bool bActive = std::find(m_vTypesForDistrParamsActive.begin(), m_vTypesForDistrParamsActive.end(), p) != m_vTypesForDistrParamsActive.end();
	SetDistrParamsActive(bActive || (m_pAnalyzer->m_nResultsType == CResultsAnalyzer::EResultType::Distribution));
}

void CResultsAnalyzerTab::UpdateTimeParams()
{
	m_bAvoidSignal = true;
	if (m_pAnalyzer->m_nRelation == CResultsAnalyzer::ERelationType::Existing)	// time points are analyzed
		ui.lineEditTimePoints->setText(QString::number(m_pAnalyzer->m_vTimePoints.size()));
	else																		// intervals are analyzed
		ui.lineEditTimePoints->setText(QString::number(m_pAnalyzer->m_vTimePoints.size() - 1));
	ui.lineEditTimeStep->setEnabled(!m_pAnalyzer->m_bOnlySavedTP);
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::SetWindowTitle(const QString& _sTitle)
{
	this->setWindowTitle(_sTitle);
}

void CResultsAnalyzerTab::SetupGeometryCombo()
{
	m_bAvoidSignal = true;
	ui.comboBoxGeometry->clear();
	for (unsigned i = 0; i < m_pSystemStructure->GeometriesNumber(); ++i)
		ui.comboBoxGeometry->insertItem(i, QString::fromStdString(m_pSystemStructure->Geometry(i)->Name()));
	if (m_pSystemStructure->GeometriesNumber() > 0)
		ui.comboBoxGeometry->setCurrentIndex(0);
	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::SetStatusText(const QString& _sText)
{
	ui.statusLabel->setText(_sText);
}

void CResultsAnalyzerTab::SetComponentActive(bool _bActive)
{
	ui.frameComponent->setEnabled(_bActive);
}

void CResultsAnalyzerTab::SetDistrParamsActive(bool _bActive)
{
	ui.frameDistrParams->setEnabled(_bActive);
}

void CResultsAnalyzerTab::UpdateWholeView()
{
	UpdateSelectedProperty();
	UpdateSelectedDistance();
	UpdateSelectedComponent();
	UpdateSelectedRelation();
	UpdateSelectedGeometry();
	UpdateDistance();
	UpdateTime();
	UpdateDistrParams();
	UpdateDistanceVisibility();
	UpdateComponentActivity();
	UpdateConstraints();
}

void CResultsAnalyzerTab::ExportDataPressed()
{
	if (m_pAnalyzer->GetCurrentStatus() == CResultsAnalyzer::EStatus::Idle)
	{
		if (ui.listProperty->selectedItems().size() > 1)
		{
			CResultsAnalyzer::VPropertyType propertyTypes;
			for (const auto& iProperty : ui.listProperty->selectedItems())
				propertyTypes.push_back(static_cast<CResultsAnalyzer::EPropertyType>(iProperty->data(Qt::UserRole).toUInt()));
			m_pAnalyzer->SetPropertyType(propertyTypes);
		}
		ExportData();
	}
	else if (m_pAnalyzer->GetCurrentStatus() == CResultsAnalyzer::EStatus::Runned)
		if (m_pAnalyzerThread)
			m_pAnalyzerThread->StopAnalyzing();
}

void CResultsAnalyzerTab::ExportData()
{
	QString sFileName = QFileDialog::getSaveFileName(this, tr("Export data"), QString::fromStdString(m_pSystemStructure->GetFileName()) + ".csv", tr("Text files (*.csv);;All files (*.*);; Dat files(*.dat);;"));
	if (sFileName.isEmpty())
	{
		SetStatusText("");
		return;
	}

	if (!IsFileWritable(sFileName))
	{
		QMessageBox::warning(this, "Writing error", "Unable to write - selected file is not writable");
		SetStatusText("");
		return;
	}

	SetStatusText("Exporting started. Please wait...");
	ui.progressBarExporting->setValue(0);
	m_pConstraintsEditorTab->SetWidgetsEnabled(false);
	ui.framePropertiesTab->setEnabled(false);
	ui.pushButtonCancel->setEnabled(false);
	ui.pushButtonExport->setText("Stop");
	emit DisableOpenGLView();

	m_pAnalyzerThread = new CAnalyzerThread(m_pAnalyzer);
	connect(m_pAnalyzerThread, SIGNAL(Finished()), this, SLOT(ExportFinished()));
	m_pAnalyzerThread->Run(sFileName);

	m_UpdateTimer.start(100);
}

void CResultsAnalyzerTab::NewPropertySelected(int n_Row)
{
	if (m_bAvoidSignal) return;

	unsigned propertyId = ui.listProperty->item(n_Row)->data(Qt::UserRole).toUInt();
	m_pAnalyzer->SetPropertyType(static_cast<CResultsAnalyzer::EPropertyType>(propertyId));

	UpdateResultsTypeActivity();
	UpdateDistrParamsActivity();
	UpdateDistanceVisibility();
	UpdateComponentActivity();
}

void CResultsAnalyzerTab::SetMultiplePropertySelection(bool _allow) const
{
	ui.listProperty->setMaximumHeight(ui.listProperty->sizeHintForRow(0) * 6 + 2 * ui.listProperty->frameWidth());
	if (!_allow)
		ui.listProperty->setSelectionMode(QAbstractItemView::SingleSelection);
}

void CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType _property, const QString& _rowNameComboBox, const QString& _sToolTip)
{
	ui.listProperty->addItem(_rowNameComboBox);
	ui.listProperty->item(ui.listProperty->count() - 1)->setData(Qt::UserRole, E2I(_property));
	ui.listProperty->item(ui.listProperty->count() - 1)->setToolTip(_sToolTip);
}

void CResultsAnalyzerTab::NewResultsTypeSelected(bool _bChecked)
{
	if (!_bChecked)	return;
	if (m_bAvoidSignal) return;

	if (ui.radioButtonDistribution->isChecked())
		m_pAnalyzer->SetResultsType(CResultsAnalyzer::EResultType::Distribution);
	else if (ui.radioButtonAverage->isChecked())
		m_pAnalyzer->SetResultsType(CResultsAnalyzer::EResultType::Average);
	else if (ui.radioButtonMaximum->isChecked())
		m_pAnalyzer->SetResultsType(CResultsAnalyzer::EResultType::Maximum);
	else if (ui.radioButtonMinimum->isChecked())
		m_pAnalyzer->SetResultsType(CResultsAnalyzer::EResultType::Minimum);
	UpdateDistrParamsActivity();
}

void CResultsAnalyzerTab::NewDistanceTypeSelected(bool _bChecked)
{
	if (!_bChecked)	return;
	if (m_bAvoidSignal) return;

	if (ui.radioButtonToPoint->isChecked())
		m_pAnalyzer->SetDistanceType(CResultsAnalyzer::EDistanceType::ToPoint);
	else
		m_pAnalyzer->SetDistanceType(CResultsAnalyzer::EDistanceType::ToLine);
	SetPoint2Visible(ui.radioButtonToLine->isChecked());

	UpdateComponentActivity();
}

void CResultsAnalyzerTab::NewComponentSelected(bool _bChecked)
{
	if (!_bChecked)	return;
	if (m_bAvoidSignal) return;

	if (ui.radioButtonTotal->isChecked())
		m_pAnalyzer->SetVectorComponent(CResultsAnalyzer::EVectorComponent::Total);
	else if (ui.radioButtonX->isChecked())
		m_pAnalyzer->SetVectorComponent(CResultsAnalyzer::EVectorComponent::X);
	else if (ui.radioButtonY->isChecked())
		m_pAnalyzer->SetVectorComponent(CResultsAnalyzer::EVectorComponent::Y);
	else if (ui.radioButtonZ->isChecked())
		m_pAnalyzer->SetVectorComponent(CResultsAnalyzer::EVectorComponent::Z);
}

void CResultsAnalyzerTab::NewRelationSelected(bool _bChecked)
{
	if (!_bChecked)	return;
	if (m_bAvoidSignal) return;

	if (ui.radioButtonNumberRelated->isChecked())
		m_pAnalyzer->SetRelatonType(CResultsAnalyzer::ERelationType::Existing);
	else if (ui.radioButtonFrequencyRelated->isChecked())
		m_pAnalyzer->SetRelatonType(CResultsAnalyzer::ERelationType::Appeared);
	UpdateTimeParams();
}

void CResultsAnalyzerTab::NewCollisionsTypeSelected(bool _bChecked)
{
	if (!_bChecked)	return;
	if (m_bAvoidSignal) return;

	if (ui.radioButtonPP->isChecked())
		m_pAnalyzer->SetCollisionType(CResultsAnalyzer::ECollisionType::ParticleParticle);
	else if (ui.radioButtonPW->isChecked())
		m_pAnalyzer->SetCollisionType(CResultsAnalyzer::ECollisionType::ParticleWall);
	UpdateSelectedCollisionsType();
}

void CResultsAnalyzerTab::NewGeometrySelected(int _nIndex)
{
	if (_nIndex < 0) return;
	if (m_bAvoidSignal) return;

	m_pAnalyzer->SetGeometryIndex(_nIndex);
}

void CResultsAnalyzerTab::NewDataPointsSet()
{
	if (m_bAvoidSignal) return;

	m_pAnalyzer->SetPoint1(CVector3(ui.lineEditX1->text().toDouble(), ui.lineEditY1->text().toDouble(), ui.lineEditZ1->text().toDouble()));
	m_pAnalyzer->SetPoint2(CVector3(ui.lineEditX2->text().toDouble(), ui.lineEditY2->text().toDouble(), ui.lineEditZ2->text().toDouble()));
}

void CResultsAnalyzerTab::NewTimeSet()
{
	if (m_bAvoidSignal) return;
	m_bAvoidSignal = true;

	double dMin = ui.lineEditTimeFrom->text().toDouble();
	double dMax = ui.lineEditTimeTo->text().toDouble();
	double dStep = ui.lineEditTimeStep->text().toDouble();
	bool bSaved = ui.radioButtonTimeSaved->isChecked();
	m_pAnalyzer->SetTime(dMin, dMax, dStep, bSaved);
	UpdateTimeParams();

	m_bAvoidSignal = false;
}

void CResultsAnalyzerTab::NewDistrParamSet()
{
	if (m_bAvoidSignal) return;

	double dMin = ui.lineEditDistrMin->text().toDouble();
	double dMax = ui.lineEditDistrMax->text().toDouble();
	unsigned nSteps = ui.lineEditDistrNClasses->text().toUInt();
	m_pAnalyzer->SetProperty(dMin, dMax, nSteps);
}

void CResultsAnalyzerTab::setVisible(bool _bVisible)
{
	QDialog::setVisible(_bVisible);
	if (_bVisible)
	{
		UpdateWholeView();
		if (m_size.isEmpty())
		{
			adjustSize();
			m_size = size();
		}
	}
}

void CResultsAnalyzerTab::CloseDialog(int _nResult)
{
	QDialog::done(_nResult);
}

void CResultsAnalyzerTab::UpdateExportStatistics()
{
	int nProgress = (int)m_pAnalyzer->GetExportProgress();
	ui.progressBarExporting->setValue(nProgress);
	SetStatusText(ss2qs(m_pAnalyzer->GetStatusDescription()));
}

void CResultsAnalyzerTab::ExportFinished()
{
	m_UpdateTimer.stop();
	m_pAnalyzerThread->Stop();
	delete m_pAnalyzerThread;
	m_pAnalyzerThread = nullptr;
	ui.progressBarExporting->setValue(100);
	if (!m_pAnalyzer->IsError())
		SetStatusText("Export finished.");
	else
		SetStatusText("Export failed: " + ss2qs(m_pAnalyzer->GetStatusDescription()));
	m_pConstraintsEditorTab->SetWidgetsEnabled(true);
	ui.framePropertiesTab->setEnabled(true);
	ui.pushButtonCancel->setEnabled(true);
	ui.pushButtonExport->setText("Export");
	emit EnableOpenGLView();
}