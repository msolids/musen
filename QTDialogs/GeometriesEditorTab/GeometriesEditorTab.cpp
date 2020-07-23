/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesEditorTab.h"
#include "qtOperations.h"
#include <QMenu>
#include <QMessageBox>

CGeometriesEditorTab::CGeometriesEditorTab( QWidget *parent ): CMusenDialog(parent)
{
	ui.setupUi(this);
	m_bAvoidSignal = false;

	setContextMenuPolicy(Qt::CustomContextMenu);
	ui.timeDepValues->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	QMenu* pGeometriesMenu = new QMenu(this);
	QMenu* pSubmenuReal = new QMenu("Real geometry", this);
	QMenu* pSubmenuVirt = new QMenu("Analysis volume", this);
	pGeometriesMenu->addMenu(pSubmenuReal);
	pGeometriesMenu->addMenu(pSubmenuVirt);
	ui.addPushButton->setMenu(pGeometriesMenu);

	m_pSubmenuRealStd = new QMenu("Standard", this);
	m_pSubmenuRealLib = new QMenu("From current database", this);
	pSubmenuReal->addMenu(m_pSubmenuRealStd);
	pSubmenuReal->addMenu(m_pSubmenuRealLib);

	m_pSubmenuVirtStd = new QMenu("Standard", this);
	m_pSubmenuVirtLib = new QMenu("From current database", this);
	pSubmenuVirt->addMenu(m_pSubmenuVirtStd);
	pSubmenuVirt->addMenu(m_pSubmenuVirtLib);

	m_pMapperRealStd = new QSignalMapper(this);
	m_pMapperRealLib = new QSignalMapper(this);
	m_pMapperVirtStd = new QSignalMapper(this);
	m_pMapperVirtLib = new QSignalMapper(this);

	InitializeConnections();

	m_sHelpFileName = "Users Guide/Geometries Editor.pdf";
}

void CGeometriesEditorTab::InitializeConnections()
{
	connect(m_pMapperRealStd, static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped), this, &CGeometriesEditorTab::AddGeometryRealStd);
	connect(m_pMapperRealLib, static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped), this, &CGeometriesEditorTab::AddGeometryRealLib);
	connect(m_pMapperVirtStd, static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped), this, &CGeometriesEditorTab::AddGeometryVirtStd);
	connect(m_pMapperVirtLib, static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped), this, &CGeometriesEditorTab::AddGeometryVirtLib);
	connect(ui.deleteGeometry, &QPushButton::clicked,				this, &CGeometriesEditorTab::RemoveGeometry);
	connect(ui.geometriesList, &QListWidget::itemChanged,			this, &CGeometriesEditorTab::GeometryNameChanged);
	connect(ui.geometriesList, &QListWidget::currentItemChanged,	this, &CGeometriesEditorTab::UpdateSelectedGeometryProperties);

	connect(ui.materialsCombo,	static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &CGeometriesEditorTab::SetMaterial);
	connect(ui.widgetColor,		&CColorView::ColorChanged,	this, &CGeometriesEditorTab::GeometryColorChanged);
	connect(ui.scaleButton,		&QPushButton::clicked,		this, &CGeometriesEditorTab::ScaleGeometry);
	connect(ui.coordX, &QLineEdit::editingFinished, 		this, &CGeometriesEditorTab::MoveGeometry);
	connect(ui.coordY, &QLineEdit::editingFinished, this, &CGeometriesEditorTab::MoveGeometry);
	connect(ui.coordZ, &QLineEdit::editingFinished, this, &CGeometriesEditorTab::MoveGeometry);
	connect(ui.rotateButton,	&QPushButton::clicked,		this, &CGeometriesEditorTab::RotateGeometry);
	connect(ui.shiftUpwards,	&QPushButton::clicked,		this, &CGeometriesEditorTab::ShiftGeometryUpwards);
	connect(ui.shiftDownwards,	&QPushButton::clicked,		this, &CGeometriesEditorTab::ShiftGeometryDownwards);

	connect(ui.param1Value,		&QLineEdit::editingFinished,	this, &CGeometriesEditorTab::GeometryParamsChanged);
	connect(ui.param2Value,		&QLineEdit::editingFinished,	this, &CGeometriesEditorTab::GeometryParamsChanged);
	connect(ui.param3Value,		&QLineEdit::editingFinished,	this, &CGeometriesEditorTab::GeometryParamsChanged);
	connect(ui.sliderQuality,	&QSlider::valueChanged,			this, &CGeometriesEditorTab::GeometryParamsChanged);

	connect(ui.rotateAroundCenter, &QCheckBox::clicked, this, &CGeometriesEditorTab::SetFreeMotion);

	connect(ui.freeMotionX,	&QCheckBox::clicked,			this, &CGeometriesEditorTab::SetFreeMotion);
	connect(ui.freeMotionY, &QCheckBox::clicked,			this, &CGeometriesEditorTab::SetFreeMotion);
	connect(ui.freeMotionZ, &QCheckBox::clicked,			this, &CGeometriesEditorTab::SetFreeMotion);
	connect(ui.objectMass,	&QLineEdit::editingFinished,	this, &CGeometriesEditorTab::SetFreeMotion);

	connect(ui.buttonAddTP,		&QPushButton::clicked, this, &CGeometriesEditorTab::AddTimePoint);
	connect(ui.buttonRemoveTP,	&QPushButton::clicked, this, &CGeometriesEditorTab::RemoveTimePoints);

	connect(ui.timeDepValues, &CQtTable::itemChanged, this, &CGeometriesEditorTab::SetTDVelocities);
	connect(this, &CGeometriesEditorTab::customContextMenuRequested, this, &CGeometriesEditorTab::ShowContextMenu);

	connect(ui.forceDepVelRadio, &QRadioButton::clicked, this, &CGeometriesEditorTab::SetTDVelocities);
	connect(ui.timeDepVelRadio, &QRadioButton::clicked, this, &CGeometriesEditorTab::SetTDVelocities);
}

void CGeometriesEditorTab::UpdateWholeView()
{
	UpdateAddButton();
	UpdateHeaders();
	UpdateGeometriesList();
	UpdateSelectedGeometryProperties();
}

void CGeometriesEditorTab::keyPressEvent(QKeyEvent *event)
{
	switch (event->key())
	{
	case Qt::Key_Delete:
		if (ui.geometriesList->hasFocus())	RemoveGeometry();
		break;
	default: CMusenDialog::keyPressEvent(event);
	}
}

void CGeometriesEditorTab::UpdateAddButton()
{
	m_pSubmenuRealStd->clear();
	m_pSubmenuRealLib->clear();
	m_pSubmenuVirtStd->clear();
	m_pSubmenuVirtLib->clear();

	m_pMapperRealStd->removeMappings(this);
	m_pMapperRealLib->removeMappings(this);
	m_pMapperVirtStd->removeMappings(this);
	m_pMapperVirtLib->removeMappings(this);

	for (size_t i = 0; i < m_pGeometriesDB->GetGeometriesNumber(); ++i)
	{
		QAction* pRealAction = new QAction(ss2qs(m_pGeometriesDB->GetGeometry(i)->sName), this);
		m_pSubmenuRealLib->addAction(pRealAction);
		connect(pRealAction, &QAction::triggered, m_pMapperRealLib, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
		m_pMapperRealLib->setMapping(pRealAction, static_cast<int>(i));

		QAction* pVirtAction = new QAction(ss2qs(m_pGeometriesDB->GetGeometry(i)->sName), this);
		m_pSubmenuVirtLib->addAction(pVirtAction);
		connect(pVirtAction, &QAction::triggered, m_pMapperVirtLib, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
		m_pMapperVirtLib->setMapping(pVirtAction, static_cast<int>(i));
	}

	for (size_t i = 0; i < CAnalysisVolume::AllVolumeTypesNames().size() - 1; ++i)
	{
		QAction* pRealAction = new QAction(ss2qs(CAnalysisVolume::AllVolumeTypesNames()[i]), this);
		m_pSubmenuRealStd->addAction(pRealAction);
		connect(pRealAction, &QAction::triggered, m_pMapperRealStd, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
		m_pMapperRealStd->setMapping(pRealAction, static_cast<int>(CAnalysisVolume::AllVolumeTypes()[i]));

		QAction* pVirtAction = new QAction(ss2qs(CAnalysisVolume::AllVolumeTypesNames()[i]), this);
		m_pSubmenuVirtStd->addAction(pVirtAction);
		connect(pVirtAction, &QAction::triggered, m_pMapperVirtStd, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
		m_pMapperVirtStd->setMapping(pVirtAction, static_cast<int>(CAnalysisVolume::AllVolumeTypes()[i]));
	}
}

void CGeometriesEditorTab::UpdateHeaders()
{
	ShowConvLabel(ui.centerLabel, "Center X:Y:Z", EUnitType::LENGTH);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(0), "Time", EUnitType::TIME);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(1), "Vx", EUnitType::VELOCITY);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(2), "Vy", EUnitType::VELOCITY);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(3), "Vz", EUnitType::VELOCITY);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(4), "RotX", EUnitType::LENGTH);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(5), "RotY", EUnitType::LENGTH);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(6), "RotZ", EUnitType::LENGTH);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(7), "Wx", EUnitType::ANGULAR_VELOCITY);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(8), "Wy", EUnitType::ANGULAR_VELOCITY);
	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(9), "Wz", EUnitType::ANGULAR_VELOCITY);
	ShowConvLabel(ui.massLabel, "Mass", EUnitType::MASS);
}

void CGeometriesEditorTab::UpdateGeometriesList()
{
	m_bAvoidSignal = true;
	int iOld = ui.geometriesList->currentRow();
	ui.geometriesList->clear();
	for (int i = 0; i < static_cast<int>(m_pSystemStructure->GetGeometriesNumber()); ++i)
	{
		ui.geometriesList->insertItem(i, ss2qs(m_pSystemStructure->GetGeometry(i)->sName));
		ui.geometriesList->item(i)->setFlags(ui.geometriesList->item(i)->flags() | Qt::ItemIsEditable);
	}
	int nOffset = static_cast<int>(m_pSystemStructure->GetGeometriesNumber());
	for (int i = 0; i < static_cast<int>(m_pSystemStructure->GetAnalysisVolumesNumber()); ++i)
	{
		ui.geometriesList->insertItem(i + nOffset, ss2qs(m_pSystemStructure->GetAnalysisVolume(i)->sName));
		ui.geometriesList->item(i + nOffset)->setFlags(ui.geometriesList->item(i + nOffset)->flags() | Qt::ItemIsEditable);
	}
	m_bAvoidSignal = false;

	if (ui.geometriesList->count() == 0)
		ui.geometriesList->setCurrentRow(-1);
	else if ((iOld != -1) && (iOld < ui.geometriesList->count()))
		ui.geometriesList->setCurrentRow(iOld);
	else
		ui.geometriesList->setCurrentRow(0);
}

void CGeometriesEditorTab::UpdateSelectedGeometryProperties()
{
	m_bAvoidSignal = true;
	ui.timeDepValues->clearContents();
	ui.timeDepValues->setRowCount(0);
	ui.materialsCombo->clear();
	EnableAllFields(false);

	if ((ui.geometriesList->currentRow() < 0) || (ui.geometriesList->currentRow() >= (int)m_pSystemStructure->GetGeometriesNumber() + (int)m_pSystemStructure->GetAnalysisVolumesNumber()))
	{
		ui.frameEditor->setEnabled(false);
		ui.deleteGeometry->setEnabled(false);
		m_bAvoidSignal = false;
		return;
	}

	ui.deleteGeometry->setEnabled(true);
	ui.frameEditor->setEnabled(true);

	if (ui.geometriesList->currentRow() < (int)m_pSystemStructure->GetGeometriesNumber())
		UpdateInfoAboutGeometry(ui.geometriesList->currentRow());
	else
		UpdateInfoAboutAnalysisVolume(ui.geometriesList->currentRow() - static_cast<int>(m_pSystemStructure->GetGeometriesNumber()));
	m_bAvoidSignal = false;
}

void CGeometriesEditorTab::UpdateInfoAboutGeometry(int _index)
{
	const SGeometryObject* pRealGeom = m_pSystemStructure->GetGeometry(_index);
	if (!pRealGeom) return;

	EnableAllFields(true);

	if (pRealGeom->bForceDepVel)
		ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(0), "Force", EUnitType::FORCE);
	else
		ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(0), "Time", EUnitType::TIME);

	ui.materialsCombo->insertItem(0, "");
	for (size_t i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
		ui.materialsCombo->insertItem(static_cast<int>(i + 1), ss2qs(m_pMaterialsDB->GetCompoundName(i)));
	if (!pRealGeom->vPlanes.empty())
	{
		if (const CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(pRealGeom->vPlanes.front()))
		{
			const int nIndex = m_pMaterialsDB->GetCompoundIndex(pObject->GetCompoundKey());
			if (nIndex != -1 && nIndex < ui.materialsCombo->count())
				ui.materialsCombo->setCurrentIndex(nIndex + 1);
		}
	}

	ui.widgetColor->setColor(pRealGeom->color);
	ShowConvValue( ui.coordX, ui.coordY, ui.coordZ, m_pSystemStructure->GetGeometryCenter(0, _index), EUnitType::LENGTH);
	ShowConvValue(ui.objectMass, pRealGeom->dMass, EUnitType::MASS);
	ui.freeMotionX->setChecked(pRealGeom->vFreeMotion.x);
	ui.freeMotionY->setChecked(pRealGeom->vFreeMotion.y);
	ui.freeMotionZ->setChecked(pRealGeom->vFreeMotion.z);
	if (!(pRealGeom->vFreeMotion.x || pRealGeom->vFreeMotion.y || pRealGeom->vFreeMotion.z))
		ui.objectMass->setEnabled(false);

	ui.rotateAroundCenter->setChecked(pRealGeom->bRotateAroundCenter);
	ui.forceDepVelRadio->setChecked(pRealGeom->bForceDepVel);
	ui.timeDepVelRadio->setChecked(!pRealGeom->bForceDepVel);

	for (int i = 0; i < static_cast<int>(pRealGeom->vIntervals.size()); ++i)
	{
		ui.timeDepValues->insertRow(i);
		for (int j = 0; j < ui.timeDepValues->columnCount(); ++j)
			ui.timeDepValues->setItem(i, j, new QTableWidgetItem(""));
		if (pRealGeom->bRotateAroundCenter)
			for (int j = 4; j < 7; ++j)
				ui.timeDepValues->item(i, j)->setFlags(ui.timeDepValues->item(i, j)->flags() ^ Qt::ItemIsEditable);
		if (pRealGeom->bForceDepVel)
			ShowConvValue(ui.timeDepValues->item(i, 0), pRealGeom->vIntervals[i].dCriticalValue, EUnitType::FORCE);
		else
			ShowConvValue(ui.timeDepValues->item(i, 0), pRealGeom->vIntervals[i].dCriticalValue, EUnitType::TIME);
		ShowVectorInTableRow(pRealGeom->vIntervals[i].vVel, ui.timeDepValues, i, 1, EUnitType::VELOCITY);
		ShowVectorInTableRow(pRealGeom->vIntervals[i].vRotCenter, ui.timeDepValues, i, 4, EUnitType::LENGTH);
		ShowVectorInTableRow(pRealGeom->vIntervals[i].vRotVel, ui.timeDepValues, i, 7, EUnitType::ANGULAR_VELOCITY);
	}
	if (pRealGeom->bRotateAroundCenter)
	{
		ui.timeDepValues->SetColumnBackgroundColor(InactiveTableColor(), 4);
		ui.timeDepValues->SetColumnBackgroundColor(InactiveTableColor(), 5);
		ui.timeDepValues->SetColumnBackgroundColor(InactiveTableColor(), 6);
	}

	UpdateInfo(pRealGeom->nVolumeType, "Real geometry: ", pRealGeom->vProps, pRealGeom->vPlanes.size());
}

void CGeometriesEditorTab::UpdateInfoAboutAnalysisVolume(int _index)
{
	const CAnalysisVolume* pVolume = m_pSystemStructure->GetAnalysisVolume(_index);
	if (!pVolume) return;

	ShowConvLabel(ui.timeDepValues->horizontalHeaderItem(0), "Time", EUnitType::TIME);
	ui.timeDepVelRadio->setChecked(true);
	for (int i = 0; i < static_cast<int>(pVolume->m_vIntervals.size()); ++i)
	{
		ui.timeDepValues->insertRow(i);
		for (int j = 0; j < ui.timeDepValues->columnCount(); ++j)
			ui.timeDepValues->setItem(i, j, new QTableWidgetItem(""));
		for (int j = 4; j < 7; ++j)
			ui.timeDepValues->item(i, j)->setFlags(ui.timeDepValues->item(i, j)->flags() ^ Qt::ItemIsEditable);
		ShowConvValue(ui.timeDepValues->item(i, 0), pVolume->m_vIntervals[i].dTime, EUnitType::TIME);
		ShowVectorInTableRow(pVolume->m_vIntervals[i].vVel, ui.timeDepValues, i, 1, EUnitType::VELOCITY);
		ShowVectorInTableRow(CVector3(0), ui.timeDepValues, i, 4, EUnitType::LENGTH);
		ShowVectorInTableRow(CVector3(0), ui.timeDepValues, i, 7, EUnitType::ANGULAR_VELOCITY);
	}


	ui.widgetColor->setColor(pVolume->color);
	ShowConvValue(ui.coordX, ui.coordY, ui.coordZ, pVolume->GetCenter(0), EUnitType::LENGTH);

	UpdateInfo(pVolume->nVolumeType, "Analysis volume: ", pVolume->vProps, pVolume->vTriangles.size());
}

void CGeometriesEditorTab::UpdateInfo(const EVolumeType& _volumeType, const QString& _sTypePrefix, const std::vector<double>& _vParams, size_t _nPlanesNum)
{
	switch (_volumeType)
	{
	case EVolumeType::VOLUME_BOX:
		ui.geometryType->setText(_sTypePrefix + "Box");
		ui.param1Value->setEnabled(true); ui.param2Value->setEnabled(true); ui.param3Value->setEnabled(true);
		ShowConvLabel(ui.param1Name, "Width X", EUnitType::LENGTH);		ShowConvValue(ui.param1Value, _vParams[0], EUnitType::LENGTH);
		ShowConvLabel(ui.param2Name, "Depth Y", EUnitType::LENGTH);		ShowConvValue(ui.param2Value, _vParams[1], EUnitType::LENGTH);
		ShowConvLabel(ui.param3Name, "Height Z", EUnitType::LENGTH);		ShowConvValue(ui.param3Value, _vParams[2], EUnitType::LENGTH);
		ui.sliderQuality->setEnabled(false);
		ui.sliderQuality->setRange(1, 1);
		ui.sliderQuality->setSliderPosition(1);
		break;
	case EVolumeType::VOLUME_SPHERE:
		ui.param1Value->setEnabled(true); ui.param2Value->setEnabled(false); ui.param3Value->setEnabled(false);
		ui.geometryType->setText(_sTypePrefix + "Sphere");
		ShowConvLabel(ui.param1Name, "Radius", EUnitType::LENGTH);
		if ( !_vParams.empty() )
			ShowConvValue(ui.param1Value, _vParams[0], EUnitType::LENGTH);
		ui.sliderQuality->setEnabled(true);
		ui.sliderQuality->setRange(0, 6);
		ui.sliderQuality->setSliderPosition(static_cast<int>(log(_nPlanesNum / 20) / log(4)));
		break;
	case EVolumeType::VOLUME_CYLINDER:
		ui.geometryType->setText(_sTypePrefix + "Cylinder");
		ui.param1Value->setEnabled(true); ui.param2Value->setEnabled(true); ui.param3Value->setEnabled(false);
		ShowConvLabel(ui.param1Name, "Radius", EUnitType::LENGTH);		ShowConvValue(ui.param1Value, _vParams[0], EUnitType::LENGTH);
		ShowConvLabel(ui.param2Name, "Height Z", EUnitType::LENGTH);		ShowConvValue(ui.param2Value, _vParams[1], EUnitType::LENGTH);
		ui.sliderQuality->setEnabled(true);
		ui.sliderQuality->setRange(4, 512);
		ui.sliderQuality->setSliderPosition(static_cast<int>(_nPlanesNum / 4));
		break;
	case EVolumeType::VOLUME_HOLLOW_SPHERE:
		ui.geometryType->setText(_sTypePrefix + "Hollow sphere");
		ui.param1Value->setEnabled(true); ui.param2Value->setEnabled(true); ui.param3Value->setEnabled(false);
		ShowConvLabel(ui.param1Name, "Radius", EUnitType::LENGTH);		ShowConvValue(ui.param1Value, _vParams[0], EUnitType::LENGTH);
		ShowConvLabel(ui.param2Name, "Inner radius", EUnitType::LENGTH); ShowConvValue(ui.param2Value, _vParams[1], EUnitType::LENGTH);
		ui.sliderQuality->setEnabled(true);
		ui.sliderQuality->setRange(0, 6);
		ui.sliderQuality->setSliderPosition(static_cast<int>(log(_nPlanesNum / 40) / log(4)));
		break;
	case EVolumeType::VOLUME_STL:
		ui.geometryType->setText(_sTypePrefix + "From database");
		ui.param1Value->setEnabled(false); ui.param2Value->setEnabled(false); ui.param3Value->setEnabled(false);
		ui.sliderQuality->setEnabled(false);
		ui.sliderQuality->setRange(1, 1);
		ui.sliderQuality->setSliderPosition(1);
		break;
	default:
		break;
	}
	ui.lineEditQuality->setText(QString::number(_nPlanesNum));
}

void CGeometriesEditorTab::EnableAllFields(bool _bEnable)
{
	ui.param1Value->setText(""); ui.param2Value->setText(""); ui.param3Value->setText("");
	ui.param1Name->setText(" "); ui.param2Name->setText(" "); ui.param3Name->setText(" ");
	ui.materialsCombo->setEnabled(_bEnable);	ui.objectMass->setEnabled(_bEnable);
	ui.rotateAroundCenter->setEnabled(_bEnable); ui.forceDepVelRadio->setEnabled(_bEnable);
	ui.freeMotionX->setEnabled(_bEnable);	ui.freeMotionY->setEnabled(_bEnable);  ui.freeMotionZ->setEnabled(_bEnable);
	ui.shiftDownwards->setEnabled(_bEnable); ui.shiftUpwards->setEnabled(_bEnable);
}

bool CGeometriesEditorTab::ConfirmRemoveAllTP()
{
	if (m_pSystemStructure->GetMaxTime() > 0)
		if (QMessageBox::question(this, "Confirmation", "This scene contains time-dependent data. All time points after 0 will be removed. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) return false;
	m_pSystemStructure->ClearAllStatesFrom(0);
	return true;
}

void CGeometriesEditorTab::ShowContextMenu(const QPoint& _pos)
{
	if (QApplication::focusWidget() == ui.timeDepValues)
	{
		QMenu myMenu;
		QAction* addNewCycle = myMenu.addAction(QIcon(":/QT_GUI/Pictures/add.png"), "Add time point");
		QAction* removeCycle = myMenu.addAction(QIcon(":/QT_GUI/Pictures/minus.png"), "Remove time point");
		QAction* selectedItem = myMenu.exec(this->mapToGlobal(_pos));
		if (selectedItem == addNewCycle)
			AddTimePoint();
		else if (selectedItem == removeCycle)
			RemoveTimePoints();
	}
}

void CGeometriesEditorTab::GeometryNameChanged(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	int nRow = ui.geometriesList->currentRow();
	QString sNewName = _pItem->text().simplified().replace(" ", "");
	if ((nRow >= 0) && (nRow < (int)m_pSystemStructure->GetGeometriesNumber()))
	{
		SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(nRow);
		if (!pGeometry) return;
		if (!sNewName.isEmpty()) pGeometry->sName = qs2ss(sNewName);
	}
	else
	{
		CAnalysisVolume* pVolume = m_pSystemStructure->GetAnalysisVolume(nRow - m_pSystemStructure->GetGeometriesNumber());
		if (!pVolume) return;
		if (!sNewName.isEmpty()) pVolume->sName = qs2ss(sNewName);
		emit ObjectsChanged();
		emit AnalysisGeometriesChanged();
	}
	UpdateWholeView();
}

void CGeometriesEditorTab::AddGeometryRealStd(int _type)
{
	if (!ConfirmRemoveAllTP()) return;

	std::vector<double> props(3);
	SVolumeType domain{ m_pSystemStructure->GetMinCoordinate(), m_pSystemStructure->GetMaxCoordinate() }; // calculate default position and size of analysis volume
	CVector3 vCenter = (domain.coordBeg + domain.coordEnd) / 2;
	for (auto& p : props)
	{
		p = Length(domain.coordEnd - domain.coordBeg) / 5;
		if (p == 0) p = 0.01; // 10 [mm] by default
	}

	m_pSystemStructure->AddGeometry(static_cast<EVolumeType>(_type), props, vCenter, CMatrix3::Diagonal());

	UpdateGeometriesList();
	ui.geometriesList->setCurrentRow((int)m_pSystemStructure->GetGeometriesNumber() - 1);
	emit ObjectsChanged();
}

void CGeometriesEditorTab::AddGeometryRealLib(int _indexDB)
{
	if (!ConfirmRemoveAllTP()) return;
	m_pSystemStructure->AddGeometry(*m_pGeometriesDB->GetGeometry(_indexDB));

	UpdateGeometriesList();
	ui.geometriesList->setCurrentRow((int)m_pSystemStructure->GetGeometriesNumber() - 1);
	emit ObjectsChanged();
}

void CGeometriesEditorTab::AddGeometryVirtStd(int _type)
{
	std::vector<double> props(3);
	SVolumeType domain{ m_pSystemStructure->GetMinCoordinate(), m_pSystemStructure->GetMaxCoordinate() }; // calculate default position and size of analysis volume
	CVector3 vCenter = (domain.coordBeg + domain.coordEnd) / 2;
	for (auto& p : props)
	{
		p = Length(domain.coordEnd - domain.coordBeg) / 5;
		if (p == 0) p = 0.01; // 10 [mm] by default
	}

	m_pSystemStructure->AddAnalysisVolume(static_cast<EVolumeType>(_type), props, vCenter, CMatrix3::Diagonal());

	emit ObjectsChanged();
	emit AnalysisGeometriesChanged();
	UpdateGeometriesList();
	ui.geometriesList->setCurrentRow((int)m_pSystemStructure->GetGeometriesNumber() + (int)m_pSystemStructure->GetAnalysisVolumesNumber() - 1);
}

void CGeometriesEditorTab::AddGeometryVirtLib(int _indexDB)
{
	m_pSystemStructure->AddAnalysisVolume(*m_pGeometriesDB->GetGeometry(_indexDB));

	emit ObjectsChanged();
	emit AnalysisGeometriesChanged();
	UpdateGeometriesList();
	ui.geometriesList->setCurrentRow((int)m_pSystemStructure->GetGeometriesNumber() + (int)m_pSystemStructure->GetAnalysisVolumesNumber() - 1);
}

void CGeometriesEditorTab::RemoveGeometry()
{
	if (QMessageBox::question(this, "Confirmation", "Delete selected geometry?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) return;
	if (ui.geometriesList->currentRow() < (int)m_pSystemStructure->GetGeometriesNumber())
		m_pSystemStructure->DeleteGeometry(ui.geometriesList->currentRow());
	else
	{
		m_pSystemStructure->DeleteAnalysisVolume(ui.geometriesList->currentRow() - m_pSystemStructure->GetGeometriesNumber());
		emit AnalysisGeometriesChanged();
	}
	UpdateWholeView();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::SetMaterial(int _iMaterial)
{
	if (m_bAvoidSignal) return;
	const CCompound* pCompound = m_pMaterialsDB->GetCompound(_iMaterial - 1);
	if (!pCompound)
	{
		ui.statusMessage->setText("Wrong compound has been specified.");
		return;
	}
	m_pSystemStructure->SetGeometryMaterial(ui.geometriesList->currentRow(), pCompound);

	ui.statusMessage->setText("New material has been specified.");
	emit ObjectsChanged();
}

void CGeometriesEditorTab::GeometryColorChanged()
{
	qreal r, g, b, f;
	ui.widgetColor->getColor().getRgbF(&r, &g, &b, &f);

	if (ui.geometriesList->currentRow() < (int)m_pSystemStructure->GetGeometriesNumber())
	{
		SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
		if(pGeometry) pGeometry->color = CColor(r, g, b, f);
	}
	else
	{
		CAnalysisVolume* pVolume = m_pSystemStructure->GetAnalysisVolume(ui.geometriesList->currentRow() - m_pSystemStructure->GetGeometriesNumber());
		if(pVolume) pVolume->color = CColor(r, g, b, f);
	}
	ui.statusMessage->setText("New color has been set.");
	UpdateSelectedGeometryProperties();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::ScaleGeometry()
{
	double dScaleFactor = ui.scalingFactor->text().toDouble();
	if (dScaleFactor <= 0)
	{
		ui.statusMessage->setText("The scaling factor must be greater than zero.");
		return;
	}

	if (ui.geometriesList->currentRow() < (int)m_pSystemStructure->GetGeometriesNumber())
	{
		if (!ConfirmRemoveAllTP()) return;
		m_pSystemStructure->ScaleGeometry(0, ui.geometriesList->currentRow(), dScaleFactor);
	}
	else
	{
		m_pSystemStructure->ScaleAnalysisVolume(ui.geometriesList->currentRow() - m_pSystemStructure->GetGeometriesNumber(), dScaleFactor);
		emit AnalysisGeometriesChanged();
	}
	ui.statusMessage->setText("The object has been scaled.");
	UpdateSelectedGeometryProperties();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::MoveGeometry()
{
	if (m_bAvoidSignal) return;
	CVector3 vNewCenter = GetConvValue(ui.coordX, ui.coordY, ui.coordZ, EUnitType::LENGTH);

	if (ui.geometriesList->currentRow() < (int)m_pSystemStructure->GetGeometriesNumber())
	{
		if (!ConfirmRemoveAllTP()) return;
		m_pSystemStructure->SetGeometryCenter(0, ui.geometriesList->currentRow(), vNewCenter);
	}
	else
	{
		m_pSystemStructure->SetAnalysisVolumeCenter(ui.geometriesList->currentRow() - m_pSystemStructure->GetGeometriesNumber(), vNewCenter);
		emit AnalysisGeometriesChanged();
	}
	ui.statusMessage->setText("The object has been moved.");
	UpdateSelectedGeometryProperties();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::RotateGeometry()
{
	CVector3 vAngle;
	if (ui.radioButtonAngleX->isChecked())
		vAngle = CVector3{ ui.lineEditAngle->text().toDouble(), 0, 0 };
	else if (ui.radioButtonAngleY->isChecked())
		vAngle = CVector3{ 0, ui.lineEditAngle->text().toDouble(), 0 };
	else if (ui.radioButtonAngleZ->isChecked())
		vAngle = CVector3{ 0, 0, ui.lineEditAngle->text().toDouble() };
	vAngle *= PI / 180;
	CMatrix3 rotationMatrix = CQuaternion(vAngle).ToRotmat();

	if (ui.geometriesList->currentRow() < (int)m_pSystemStructure->GetGeometriesNumber())
	{
		if (!ConfirmRemoveAllTP()) return;
		m_pSystemStructure->RotateGeometry(0, ui.geometriesList->currentRow(), rotationMatrix);
	}
	else
	{
		m_pSystemStructure->RotateAnalysisVolume(ui.geometriesList->currentRow() - m_pSystemStructure->GetGeometriesNumber(), rotationMatrix);
		emit AnalysisGeometriesChanged();
	}
	ui.statusMessage->setText("The object has been rotated.");
	emit ObjectsChanged();
	UpdateSelectedGeometryProperties();
}

void CGeometriesEditorTab::ShiftGeometryUpwards()
{
	SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
	if (!pGeometry) return;

	double dMinZ = m_pSystemStructure->GetMaxCoordinate(0).z;
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
	{
		CPhysicalObject* pTemp = m_pSystemStructure->GetObjectByIndex(i);
		if (!pTemp) continue;
		if (pTemp->GetObjectType() == SPHERE)
			if (dMinZ > pTemp->GetCoordinates(0).z - static_cast<CSphere*>(pTemp)->GetRadius())
				dMinZ = pTemp->GetCoordinates(0).z - static_cast<CSphere*>(pTemp)->GetRadius();
	}

	double dMaxZWall = 0;
	std::vector<CTriangularWall*> walls = m_pSystemStructure->GetGeometryWalls(ui.geometriesList->currentRow());
	for (size_t i = 0; i < walls.size(); ++i)
	{
		double dTempMax = std::max({ walls[i]->GetCoordVertex1(0).z, walls[i]->GetCoordVertex2(0).z, walls[i]->GetCoordVertex3(0).z });
		if (i == 0)
			dMaxZWall = dTempMax;
		else
			dMaxZWall = std::max(dMaxZWall, dTempMax);
	}

	CVector3 vOffset(0, 0, dMinZ - dMaxZWall);
	for (auto wall : walls)
		wall->SetPlaneCoord(0., wall->GetCoordVertex1(0) + vOffset, wall->GetCoordVertex2(0) + vOffset, wall->GetCoordVertex3(0) + vOffset);

	UpdateSelectedGeometryProperties();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::ShiftGeometryDownwards()
{
	SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
	if (!pGeometry) return;

	double dMaxZ = m_pSystemStructure->GetMinCoordinate(0).z;
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
	{
		CPhysicalObject* pTemp = m_pSystemStructure->GetObjectByIndex(i);
		if (!pTemp) continue;
		if (pTemp->GetObjectType() == SPHERE)
			if (dMaxZ < pTemp->GetCoordinates(0).z + static_cast<CSphere*>(pTemp)->GetRadius())
				dMaxZ = pTemp->GetCoordinates(0).z + static_cast<CSphere*>(pTemp)->GetRadius();
	}

	double dMinZWall = 0;
	std::vector<CTriangularWall*> walls = m_pSystemStructure->GetGeometryWalls(ui.geometriesList->currentRow());
	for (size_t i = 0; i < walls.size(); ++i)
	{
		double dTempMin = std::min({ walls[i]->GetCoordVertex1(0).z, walls[i]->GetCoordVertex2(0).z, walls[i]->GetCoordVertex3(0).z });
		if (i == 0)
			dMinZWall = dTempMin;
		else
			dMinZWall = std::min(dMinZWall, dTempMin);
	}

	CVector3 vOffset(0, 0, dMaxZ - dMinZWall);
	for (auto wall : walls)
		wall->SetPlaneCoord(0., wall->GetCoordVertex1(0) + vOffset, wall->GetCoordVertex2(0) + vOffset, wall->GetCoordVertex3(0) + vOffset);

	UpdateSelectedGeometryProperties();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::GeometryParamsChanged()
{
	if (m_bAvoidSignal) return;

	int iRow = ui.geometriesList->currentRow();
	std::vector<double> vParams{ GetConvValue(ui.param1Value, EUnitType::LENGTH), GetConvValue(ui.param2Value, EUnitType::LENGTH), GetConvValue(ui.param3Value, EUnitType::LENGTH) };

	if (iRow < static_cast<int>(m_pSystemStructure->GetGeometriesNumber()))
	{
		SGeometryObject* pOldGeometry = m_pSystemStructure->GetGeometry(iRow);
		if (!pOldGeometry) return;
		if (pOldGeometry->nVolumeType == EVolumeType::VOLUME_STL) return;
		SGeometryObject tempGeometry = *pOldGeometry;
		if (!ConfirmRemoveAllTP()) return;

		CVector3 vCenter = m_pSystemStructure->GetGeometryCenter(0, iRow);
		m_pSystemStructure->DeleteGeometry(iRow);
		SGeometryObject* pNewGeometry = m_pSystemStructure->AddGeometry(tempGeometry.nVolumeType, vParams, vCenter, tempGeometry.mRotation, ui.sliderQuality->value());
		pNewGeometry->sName = tempGeometry.sName;
		pNewGeometry->sKey = tempGeometry.sKey;
		pNewGeometry->color = tempGeometry.color;
		pNewGeometry->dMass = tempGeometry.dMass;
		pNewGeometry->vFreeMotion = tempGeometry.vFreeMotion;
		pNewGeometry->vIntervals = tempGeometry.vIntervals;
		pNewGeometry->bRotateAroundCenter = tempGeometry.bRotateAroundCenter;

		// move to the old position
		for (size_t i = 0; i < m_pSystemStructure->GetGeometriesNumber() - iRow - 1; ++i)
			m_pSystemStructure->UpGeometry(m_pSystemStructure->GetGeometriesNumber() - 1 - i);
	}
	else
	{
		CAnalysisVolume* pOldVolume = m_pSystemStructure->GetAnalysisVolume(iRow - m_pSystemStructure->GetGeometriesNumber());
		if (!pOldVolume) return;
		if (pOldVolume->nVolumeType == EVolumeType::VOLUME_STL) return;
		CAnalysisVolume tempVolume = *pOldVolume;

		m_pSystemStructure->DeleteAnalysisVolume(iRow - m_pSystemStructure->GetGeometriesNumber());
		CAnalysisVolume* pNewVolume = m_pSystemStructure->AddAnalysisVolume(tempVolume.nVolumeType, vParams, tempVolume.GetCenter(0), tempVolume.mRotation, ui.sliderQuality->value());
		pNewVolume->sName = tempVolume.sName;
		pNewVolume->sKey = tempVolume.sKey;
		pNewVolume->color = tempVolume.color;

		// move to the old position
		for (size_t i = 0; i < m_pSystemStructure->GetAnalysisVolumesNumber() - (iRow - m_pSystemStructure->GetGeometriesNumber()) - 1; ++i)
			m_pSystemStructure->UpAnalysisVolume(m_pSystemStructure->GetAnalysisVolumesNumber() - 1 - i);

		emit AnalysisGeometriesChanged();
	}
	UpdateSelectedGeometryProperties();
	emit ObjectsChanged();
}

void CGeometriesEditorTab::SetFreeMotion()
{
	if (m_bAvoidSignal) return;
	SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
	if (!pGeometry) return;
	pGeometry->dMass = GetConvValue(ui.objectMass, EUnitType::MASS);
	pGeometry->vFreeMotion.Init(0);
	if (ui.freeMotionX->isChecked()) pGeometry->vFreeMotion.x = 1;
	if (ui.freeMotionY->isChecked()) pGeometry->vFreeMotion.y = 1;
	if (ui.freeMotionZ->isChecked()) pGeometry->vFreeMotion.z = 1;
	if (pGeometry->vFreeMotion.Length() > 0)
		ui.objectMass->setEnabled(true);
	else
		ui.objectMass->setEnabled(false);

	pGeometry->bRotateAroundCenter = ui.rotateAroundCenter->isChecked();
	UpdateSelectedGeometryProperties();
}

void CGeometriesEditorTab::AddTimePoint()
{
	int iRow = ui.geometriesList->currentRow();
	if (iRow < static_cast<int>(m_pSystemStructure->GetGeometriesNumber() ))
	{
		SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
		if (!pGeometry) return;
		pGeometry->AddTimePoint();
	}
	else
	{
		CAnalysisVolume* pAnalysisVolume = m_pSystemStructure->GetAnalysisVolume(iRow - m_pSystemStructure->GetGeometriesNumber());
		if (!pAnalysisVolume) return;
		pAnalysisVolume->AddTimePoint();
	}
	UpdateSelectedGeometryProperties();
}

void CGeometriesEditorTab::RemoveTimePoints()
{
	QItemSelection selection(ui.timeDepValues->selectionModel()->selection());
	QSet<int> rows;
	foreach(const QModelIndex &index, selection.indexes())
		rows.insert(index.row());
	QList<int> rowsList = rows.values();
	qSort(rowsList);

	int iRow = ui.geometriesList->currentRow();
	if (iRow < static_cast<int>(m_pSystemStructure->GetGeometriesNumber()))
	{
		SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
		if (!pGeometry) return;
		for (int i = rowsList.count() - 1; i >= 0; --i)
			pGeometry->vIntervals.erase(pGeometry->vIntervals.begin() + rowsList[i]);
	} else
	{
		CAnalysisVolume* pAnalysisVolume = m_pSystemStructure->GetAnalysisVolume(iRow - m_pSystemStructure->GetGeometriesNumber());
		if (!pAnalysisVolume) return;
		for (int i = rowsList.count() - 1; i >= 0; --i)
			pAnalysisVolume->m_vIntervals.erase(pAnalysisVolume->m_vIntervals.begin() + rowsList[i]);

	}
	UpdateSelectedGeometryProperties();
}

void CGeometriesEditorTab::SetTDVelocities()
{
	if (m_bAvoidSignal) return;

	int iRow = ui.geometriesList->currentRow();
	if (iRow < static_cast<int>(m_pSystemStructure->GetGeometriesNumber()))
	{

		SGeometryObject* pGeometry = m_pSystemStructure->GetGeometry(ui.geometriesList->currentRow());
		if (!pGeometry) return;
		pGeometry->vIntervals.clear();
		pGeometry->bForceDepVel = !ui.timeDepVelRadio->isChecked();

		for (int i = 0; i < ui.timeDepValues->rowCount(); ++i)
		{
			double dLimitValue;
			if (pGeometry->bForceDepVel)
				dLimitValue = GetConvValue(ui.timeDepValues->item(i, 0), EUnitType::FORCE);
			else
				dLimitValue = GetConvValue(ui.timeDepValues->item(i, 0), EUnitType::TIME);

			CVector3 vVel = GetVectorFromTableRow(ui.timeDepValues, i, 1, EUnitType::VELOCITY);
			CVector3 vRotCenter = GetVectorFromTableRow(ui.timeDepValues, i, 4, EUnitType::LENGTH);
			CVector3 vRotVel = GetVectorFromTableRow(ui.timeDepValues, i, 7, EUnitType::ANGULAR_VELOCITY);
			pGeometry->vIntervals.push_back({ dLimitValue, vVel, vRotVel, vRotCenter });
		}
	}
	else
	{
		CAnalysisVolume* pAnalysisVolume = m_pSystemStructure->GetAnalysisVolume(iRow - m_pSystemStructure->GetGeometriesNumber());
		if (!pAnalysisVolume) return;
		pAnalysisVolume->m_vIntervals.clear();
		for (int i = 0; i < ui.timeDepValues->rowCount(); ++i)
			pAnalysisVolume->m_vIntervals.push_back({ GetConvValue(ui.timeDepValues->item(i, 0), EUnitType::TIME),
				GetVectorFromTableRow(ui.timeDepValues, i, 1, EUnitType::VELOCITY) });
	}
	UpdateSelectedGeometryProperties();
}