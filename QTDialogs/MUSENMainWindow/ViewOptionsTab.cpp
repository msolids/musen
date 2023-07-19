/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ViewOptionsTab.h"
#include "QtSignalBlocker.h"
#include "qtOperations.h"
#include <QKeyEvent>

CViewOptionsTab::CViewOptionsTab(CViewManager* _viewManager, CViewSettings* _viewSettings, QWidget* parent) :
	CMusenDialog(parent),
	m_viewManager{ _viewManager },
	m_viewSettings{ _viewSettings }
{
	ui.setupUi(this);
	m_dCurrentTime = 0;

	m_coloringVectors = { EColoring::ANGLE_VELOCITY, EColoring::COORDINATE, EColoring::DISPLACEMENT, EColoring::FORCE, EColoring::STRESS, EColoring::PRINCIPAL_STRESS, EColoring::VELOCITY };
	m_componentGroup.addButton(ui.radioButtonComponentL, E2I(EColorComponent::TOTAL));
	m_componentGroup.addButton(ui.radioButtonComponentX, E2I(EColorComponent::X));
	m_componentGroup.addButton(ui.radioButtonComponentY, E2I(EColorComponent::Y));
	m_componentGroup.addButton(ui.radioButtonComponentZ, E2I(EColorComponent::Z));

	m_slicingGroup.addButton(ui.radioButtonSliceX, E2I(ESlicePlane::X));
	m_slicingGroup.addButton(ui.radioButtonSliceY, E2I(ESlicePlane::Y));
	m_slicingGroup.addButton(ui.radioButtonSliceZ, E2I(ESlicePlane::Z));
	m_slicingGroup.setExclusive(true);

	ui.objectsInfoTable->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.objectsInfoTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	UpdateColoringColors();

	InitializeConnections();
}

void CViewOptionsTab::InitializeConnections() const
{
	////////// PAGE 1 - Cutting //////////

	// cutting planes check boxes
	connect(ui.enableCuttingX, &QCheckBox::toggled, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.enableCuttingY, &QCheckBox::toggled, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.enableCuttingZ, &QCheckBox::toggled, this, &CViewOptionsTab::OnCuttingChanged);

	// cutting planes edit boxes
	connect(ui.maxXEdit, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.minXEdit, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.minYEdit, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.maxYEdit, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.minZEdit, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.maxZEdit, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnCuttingChanged);

	// cutting volumes
	connect(ui.groupBoxCutByVolume, &QGroupBox::toggled,       this, &CViewOptionsTab::OnCuttingChanged);
	connect(ui.listCuttingVolumes,  &QListWidget::itemChanged, this, &CViewOptionsTab::OnCuttingChanged);

	// slicing
	connect(ui.groupBoxSlices,		&QGroupBox::toggled,				this, &CViewOptionsTab::OnSlicingChanged);
	connect(ui.spinBoxSliceCoords,  &CQtDoubleSpinBox::ValueChanged,	this, &CViewOptionsTab::OnSlicingChanged);
	connect(&m_slicingGroup, QOverload<QAbstractButton*, bool>::of(&QButtonGroup::buttonToggled), [=](QAbstractButton*, bool _checked) { if (_checked) OnSlicingChanged(); });

	////////// PAGE 2 - Visibility //////////

	// particles
	connect(ui.groupBoxShowParticles,	&QGroupBox::toggled,		this, &CViewOptionsTab::OnParticlesVisibilityChanged);
	connect(ui.listParticlesMaterials,	&QListWidget::itemChanged,	this, &CViewOptionsTab::OnParticlesVisibilityChanged);
	connect(ui.checkBoxOrientations,	&QCheckBox::toggled,		this, &CViewOptionsTab::OnParticlesVisibilityChanged);

	// bonds
	connect(ui.groupBoxShowBonds,          &QGroupBox::toggled,         this, &CViewOptionsTab::OnBondsVisibilityChanged);
	connect(ui.listBondsMaterials,         &QListWidget::itemChanged,   this, &CViewOptionsTab::OnBondsVisibilityChanged);
	connect(ui.groupBoxBrokenBonds,        &QGroupBox::toggled,         this, &CViewOptionsTab::OnBondsVisibilityChanged);
	connect(ui.lineEditTimeBegBrokenBonds, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnBondsVisibilityChanged);
	connect(ui.lineEditTimeEndBrokenBonds, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnBondsVisibilityChanged);
	connect(ui.colorBrokenBonds,           &CColorView::ColorChanged,   this, &CViewOptionsTab::OnBondsVisibilityChanged);

	// geometries
	connect(ui.groupBoxShowGeometries,	&QGroupBox::toggled,		this, &CViewOptionsTab::OnGeometriesVisibilityChanged);
	connect(ui.listGeometries,			&QListWidget::itemChanged,	this, &CViewOptionsTab::OnGeometriesVisibilityChanged);
	connect(ui.sliderOpacity,		    &QSlider::valueChanged,		this, &CViewOptionsTab::OnGeometriesVisibilityChanged);

	// volumes
	connect(ui.groupBoxShowVolumes, &QGroupBox::toggled,		this, &CViewOptionsTab::OnVolumesVisibilityChanged);
	connect(ui.listVolumes,			&QListWidget::itemChanged,	this, &CViewOptionsTab::OnVolumesVisibilityChanged);

	// check boxes
	connect(ui.checkBoxDomain,	&QCheckBox::toggled, this, &CViewOptionsTab::OnDomainVisibilityChanged);
	connect(ui.checkBoxPBC,		&QCheckBox::toggled, this, &CViewOptionsTab::OnPBCVisibilityChanged);

	////////// PAGE 3 - Coloring //////////

	// coloring parameters
	connect(&m_coloringGroup,  QOverload<QAbstractButton*, bool>::of(&QButtonGroup::buttonToggled), [=](QAbstractButton*, bool _checked) { if (_checked) OnColoringChanged(); });
	connect(&m_componentGroup, QOverload<QAbstractButton*, bool>::of(&QButtonGroup::buttonToggled), [=](QAbstractButton*, bool _checked) { if (_checked) OnColoringChanged(); });
	connect(ui.lineEditColorLimitMin, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnColoringChanged);
	connect(ui.lineEditColorLimitMax, &QLineEdit::editingFinished, this, &CViewOptionsTab::OnColoringChanged);
	connect(ui.colorMin,			  &CColorView::ColorChanged,   this, &CViewOptionsTab::OnColoringChanged);
	connect(ui.colorMid,			  &CColorView::ColorChanged,   this, &CViewOptionsTab::OnColoringChanged);
	connect(ui.colorMax,			  &CColorView::ColorChanged,   this, &CViewOptionsTab::OnColoringChanged);

	// auto coloring limits button
	connect(ui.pushButtonColorLimitsAuto, &QPushButton::clicked, this, &CViewOptionsTab::OnAutoColorLimits);
}

void CViewOptionsTab::UpdateWholeView()
{
	UpdateLabels();

	UpdateCuttingPlanes();
	UpdateCuttingVolumes();
	UpdateSlicing();

	UpdateParticlesVisibility();
	UpdateBondsVisibility();
	UpdateGeometriesVisibility();
	UpdateVolumesVisibility();
	UpdateVisibilityCheckboxes();

	UpdateSelectedObjectsInfo();

	UpdateColoringTypes();
	UpdateColoringLimits();
	UpdateColoringComponent();
}

void CViewOptionsTab::NewSceneLoaded()
{
	SelectAllParticleMaterials();
	SelectAllBondMaterials();
	SelectAllGeometries();
	SelectAllVolumes();
}

void CViewOptionsTab::UpdateMaterials() const
{
	SelectAllParticleMaterials();
	SelectAllBondMaterials();
	UpdateParticlesVisibility();
	UpdateBondsVisibility();
}

void CViewOptionsTab::UpdateGeometries() const
{
	UpdateGeometriesVisibility();
	UpdateVolumesVisibility();
}

void CViewOptionsTab::UpdateSelectedObjects() const
{
	ui.mainToolBox->setCurrentIndex(ETab::SELECTION);	// show proper tab
	UpdateSelectedObjectsInfo();						// update info
}

void CViewOptionsTab::OnCuttingChanged() const
{
	CViewSettings::SCutting cuttingSettings = m_viewSettings->Cutting();

	cuttingSettings.cutByX = ui.enableCuttingX->isChecked();
	cuttingSettings.cutByY = ui.enableCuttingY->isChecked();
	cuttingSettings.cutByZ = ui.enableCuttingZ->isChecked();

	cuttingSettings.minX = GetConvValue(ui.minXEdit, EUnitType::LENGTH);
	cuttingSettings.maxX = GetConvValue(ui.maxXEdit, EUnitType::LENGTH);
	cuttingSettings.minY = GetConvValue(ui.minYEdit, EUnitType::LENGTH);
	cuttingSettings.maxY = GetConvValue(ui.maxYEdit, EUnitType::LENGTH);
	cuttingSettings.minZ = GetConvValue(ui.minZEdit, EUnitType::LENGTH);
	cuttingSettings.maxZ = GetConvValue(ui.maxZEdit, EUnitType::LENGTH);

	cuttingSettings.cutByVolumes = ui.groupBoxCutByVolume->isChecked();
	cuttingSettings.volumes.clear();
	for (int i = 0; i < ui.listCuttingVolumes->count(); ++i)
		if (ui.listCuttingVolumes->item(i)->checkState() == Qt::Checked)
			cuttingSettings.volumes.insert(ui.listCuttingVolumes->item(i)->data(Qt::UserRole).toString().toStdString());

	m_viewSettings->Cutting(cuttingSettings);

	UpdateCuttingPlanes();
	UpdateCuttingVolumes();

	m_viewManager->UpdateAllObjects();
}

void CViewOptionsTab::OnSlicingChanged() const
{
	CViewSettings::SSlicing slicingSettings = m_viewSettings->Slicing();

	const auto plane = static_cast<ESlicePlane>(m_slicingGroup.checkedId());
	slicingSettings.active = ui.groupBoxSlices->isChecked();
	slicingSettings.plane = plane != static_cast<ESlicePlane>(-1) ? plane : ESlicePlane::NONE; // to handle NONE-case properly
	slicingSettings.coordinate = m_pUnitConverter->GetValueSI(EUnitType::LENGTH, ui.spinBoxSliceCoords->GetValue());

	m_viewSettings->Slicing(slicingSettings);
	m_viewManager->UpdateAllObjects();
}

void CViewOptionsTab::OnParticlesVisibilityChanged() const
{
	// visibility of all particles
	CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	visibilitySettings.particles = ui.groupBoxShowParticles->isChecked();
	m_viewSettings->Visibility(visibilitySettings);

	// orientations
	visibilitySettings.orientations = ui.checkBoxOrientations->isChecked();
	m_viewSettings->Visibility(visibilitySettings);

	// visibility of particles' materials
	auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();
	visiblePartMaterials.clear();
	for (int i = 0; i < ui.listParticlesMaterials->count(); ++i)
		if (ui.listParticlesMaterials->item(i)->checkState() == Qt::Checked)
			visiblePartMaterials.insert(ui.listParticlesMaterials->item(i)->data(Qt::UserRole).toString().toStdString());
	m_viewSettings->VisiblePartMaterials(visiblePartMaterials);

	// update
	m_viewManager->UpdateParticles();
	UpdateParticlesVisibility();
}

void CViewOptionsTab::OnBondsVisibilityChanged() const
{
	// visibility of all bonds
	CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	visibilitySettings.bonds = ui.groupBoxShowBonds->isChecked();
	m_viewSettings->Visibility(visibilitySettings);

	// visibility of bonds' materials
	auto visibleBondMaterials = m_viewSettings->VisibleBondMaterials();
	visibleBondMaterials.clear();
	for (int i = 0; i < ui.listBondsMaterials->count(); ++i)
		if (ui.listBondsMaterials->item(i)->checkState() == Qt::Checked)
			visibleBondMaterials.insert(ui.listBondsMaterials->item(i)->data(Qt::UserRole).toString().toStdString());
	m_viewSettings->VisibleBondMaterials(visibleBondMaterials);

	// visibility of broken bonds
	CViewSettings::SBrokenBonds brokenBondsSettings = m_viewSettings->BrokenBonds();
	brokenBondsSettings.show = ui.groupBoxBrokenBonds->isChecked();
	brokenBondsSettings.startTime = GetConvValue(ui.lineEditTimeBegBrokenBonds, EUnitType::TIME);
	brokenBondsSettings.endTime = GetConvValue(ui.lineEditTimeEndBrokenBonds, EUnitType::TIME);
	brokenBondsSettings.color = ui.colorBrokenBonds->getColor();
	m_viewSettings->BrokenBonds(brokenBondsSettings);

	// update
	m_viewManager->UpdateBonds();
	UpdateBondsVisibility();
}

void CViewOptionsTab::OnGeometriesVisibilityChanged() const
{
	// visibility of all geometries
	CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	visibilitySettings.geometries = ui.groupBoxShowGeometries->isChecked();
	m_viewSettings->Visibility(visibilitySettings);

	// visibility of separate geometries
	auto visibleGeometries = m_viewSettings->VisibleGeometries();
	visibleGeometries.clear();
	for (int i = 0; i < ui.listGeometries->count(); ++i)
		if (ui.listGeometries->item(i)->checkState() == Qt::Checked)
			visibleGeometries.insert(ui.listGeometries->item(i)->data(Qt::UserRole).toString().toStdString());
	m_viewSettings->VisibleGeometries(visibleGeometries);

	// transparency
	m_viewSettings->GeometriesTransparency(1.0f - ui.sliderOpacity->sliderPosition() / 10.0f);

	// update
	m_viewManager->UpdateGeometries();
	UpdateGeometriesVisibility();
}

void CViewOptionsTab::OnVolumesVisibilityChanged() const
{
	// visibility of all volumes
	CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	visibilitySettings.volumes = ui.groupBoxShowVolumes->isChecked();
	m_viewSettings->Visibility(visibilitySettings);

	// visibility of separate volumes
	auto visibleVolumes = m_viewSettings->VisibleVolumes();
	visibleVolumes.clear();
	for (int i = 0; i < ui.listVolumes->count(); ++i)
		if (ui.listVolumes->item(i)->checkState() == Qt::Checked)
			visibleVolumes.insert(ui.listVolumes->item(i)->data(Qt::UserRole).toString().toStdString());
	m_viewSettings->VisibleVolumes(visibleVolumes);

	// update
	m_viewManager->UpdateVolumes();
	UpdateVolumesVisibility();
}

void CViewOptionsTab::OnDomainVisibilityChanged() const
{
	CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	visibilitySettings.domain = ui.checkBoxDomain->isChecked();
	m_viewSettings->Visibility(visibilitySettings);
	m_viewManager->UpdateDomain();
}

void CViewOptionsTab::OnPBCVisibilityChanged() const
{
	CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	visibilitySettings.pbc = ui.checkBoxPBC->isChecked();
	m_viewSettings->Visibility(visibilitySettings);
	m_viewManager->UpdatePBC();
}

void CViewOptionsTab::OnColoringChanged() const
{
	CViewSettings::SColoring colorSettings = m_viewSettings->Coloring();
	colorSettings.type = static_cast<EColoring>(m_coloringGroup.checkedId());
	const EUnitType units = static_cast<EUnitType>(m_coloringGroup.button(E2I(colorSettings.type))->property("units").toUInt()); // get measurement units from property of the corresponding radio button
	colorSettings.minValue = GetConvValue(ui.lineEditColorLimitMin, units);
	colorSettings.maxValue = GetConvValue(ui.lineEditColorLimitMax, units);
	colorSettings.minDisplayValue = ui.lineEditColorLimitMin->text().toDouble();
	colorSettings.maxDisplayValue = ui.lineEditColorLimitMax->text().toDouble();
	colorSettings.component = static_cast<EColorComponent>(m_componentGroup.checkedId());
	colorSettings.minColor = ui.colorMin->getColor();
	colorSettings.midColor = ui.colorMid->getColor();
	colorSettings.maxColor = ui.colorMax->getColor();
	m_viewSettings->Coloring(colorSettings);

	UpdateColoringLimits();
	UpdateColoringComponent();

	m_viewManager->UpdateColors();
}

void CViewOptionsTab::OnAutoColorLimits() const
{
	const std::vector<double> values = m_viewManager->GetColoringValues(m_pSystemStructure->GetAllSpheres(m_dCurrentTime), m_pSystemStructure->GetAllSolidBonds(m_dCurrentTime));

	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.lineEditColorLimitMin, ui.lineEditColorLimitMax });
	const EUnitType units = static_cast<EUnitType>(m_coloringGroup.button(m_coloringGroup.checkedId())->property("units").toUInt()); // get measurement units from property of the corresponding radio button
	ui.lineEditColorLimitMin->setText(QString::number(m_pUnitConverter->GetValue(units, VectorMin(values))));
	ui.lineEditColorLimitMax->setText(QString::number(m_pUnitConverter->GetValue(units, VectorMax(values))));

	// update
	OnColoringChanged();
}

void CViewOptionsTab::keyPressEvent(QKeyEvent *e)
{
	if(e->key() != Qt::Key_Escape)
		CMusenDialog::keyPressEvent(e);
}

void CViewOptionsTab::UpdateCuttingPlanes() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.enableCuttingX, ui.enableCuttingY, ui.enableCuttingZ, ui.minXEdit, ui.maxXEdit, ui.minYEdit, ui.maxYEdit, ui.minZEdit, ui.maxZEdit });

	const CViewSettings::SCutting cuttingSettings = m_viewSettings->Cutting();
	ui.enableCuttingX->setChecked(cuttingSettings.cutByX);
	ui.enableCuttingY->setChecked(cuttingSettings.cutByY);
	ui.enableCuttingZ->setChecked(cuttingSettings.cutByZ);
	ui.minXEdit->setEnabled(cuttingSettings.cutByX);
	ui.maxXEdit->setEnabled(cuttingSettings.cutByX);
	ui.minYEdit->setEnabled(cuttingSettings.cutByY);
	ui.maxYEdit->setEnabled(cuttingSettings.cutByY);
	ui.minZEdit->setEnabled(cuttingSettings.cutByZ);
	ui.maxZEdit->setEnabled(cuttingSettings.cutByZ);
	ShowConvValue(ui.minXEdit, cuttingSettings.minX, EUnitType::LENGTH);
	ShowConvValue(ui.maxXEdit, cuttingSettings.maxX, EUnitType::LENGTH);
	ShowConvValue(ui.minYEdit, cuttingSettings.minY, EUnitType::LENGTH);
	ShowConvValue(ui.maxYEdit, cuttingSettings.maxY, EUnitType::LENGTH);
	ShowConvValue(ui.minZEdit, cuttingSettings.minZ, EUnitType::LENGTH);
	ShowConvValue(ui.maxZEdit, cuttingSettings.maxZ, EUnitType::LENGTH);
}

void CViewOptionsTab::UpdateCuttingVolumes() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.groupBoxCutByVolume, ui.listCuttingVolumes });

	const CViewSettings::SCutting cuttingSettings = m_viewSettings->Cutting();

	ui.groupBoxCutByVolume->setChecked(cuttingSettings.cutByVolumes);
	ui.listCuttingVolumes->setEnabled(cuttingSettings.cutByVolumes);
	ui.listCuttingVolumes->clear();

	for (const auto& volume : m_pSystemStructure->AllAnalysisVolumes())
	{
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(volume->Name()), ui.listCuttingVolumes);
		item->setData(Qt::UserRole, QString::fromStdString(volume->Key()));
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(SetContains(cuttingSettings.volumes, volume->Key()) ? Qt::Checked : Qt::Unchecked);
		ui.listCuttingVolumes->addItem(item);
	}
}

void CViewOptionsTab::UpdateSlicing() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.groupBoxSlices, ui.radioButtonSliceX, ui.radioButtonSliceY, ui.radioButtonSliceZ, ui.spinBoxSliceCoords });

	const CViewSettings::SSlicing slicingSettings = m_viewSettings->Slicing();

	ui.groupBoxSlices->setChecked(slicingSettings.active);
	ui.radioButtonSliceX->setChecked(slicingSettings.plane == ESlicePlane::X);
	ui.radioButtonSliceY->setChecked(slicingSettings.plane == ESlicePlane::Y);
	ui.radioButtonSliceZ->setChecked(slicingSettings.plane == ESlicePlane::Z);
	ui.spinBoxSliceCoords->SetValue(m_pUnitConverter->GetValue(EUnitType::LENGTH, slicingSettings.coordinate));
}

void CViewOptionsTab::UpdateParticlesVisibility() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.groupBoxShowParticles, ui.listParticlesMaterials, ui.checkBoxOrientations });

	const CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	const auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();

	// visibility of particles
	ui.groupBoxShowParticles->setChecked(visibilitySettings.particles);

	// materials
	ui.listParticlesMaterials->clear();
	for (const auto& key : m_pSystemStructure->GetAllParticlesCompounds())
	{
		const CCompound* compound = m_pMaterialsDB->GetCompound(key);
		QListWidgetItem* item = new QListWidgetItem(compound ? QString::fromStdString(compound->GetName()) : "unknown", ui.listParticlesMaterials);
		item->setData(Qt::UserRole, QString::fromStdString(key));
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(SetContains(visiblePartMaterials, key) ? Qt::Checked : Qt::Unchecked);
		ui.listParticlesMaterials->addItem(item);
	}

	// orientations
	ui.checkBoxOrientations->setChecked(visibilitySettings.orientations);

}

void CViewOptionsTab::UpdateBondsVisibility() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.groupBoxShowBonds, ui.listBondsMaterials, ui.colorBrokenBonds });

	const CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	const auto visibleBondMaterials = m_viewSettings->VisibleBondMaterials();

	// visibility of bonds
	ui.groupBoxShowBonds->setChecked(visibilitySettings.bonds);

	// materials
	ui.listBondsMaterials->clear();
	for (const auto& key : m_pSystemStructure->GetAllBondsCompounds())
	{
		const CCompound* compound = m_pMaterialsDB->GetCompound(key);
		QListWidgetItem* item = new QListWidgetItem(compound ? QString::fromStdString(compound->GetName()) : "unknown", ui.listBondsMaterials);
		item->setData(Qt::UserRole, QString::fromStdString(key));
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(SetContains(visibleBondMaterials, key) ? Qt::Checked : Qt::Unchecked);
		ui.listBondsMaterials->addItem(item);
	}

	// broken bonds
	const CViewSettings::SBrokenBonds brokenBondsSettings = m_viewSettings->BrokenBonds();
	ui.groupBoxBrokenBonds->setChecked(brokenBondsSettings.show);
	ShowConvValue(ui.lineEditTimeBegBrokenBonds, brokenBondsSettings.startTime, EUnitType::TIME);
	ShowConvValue(ui.lineEditTimeEndBrokenBonds, brokenBondsSettings.endTime,	EUnitType::TIME);
	ui.colorBrokenBonds->SetColor(brokenBondsSettings.color);
}

void CViewOptionsTab::UpdateGeometriesVisibility() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.groupBoxShowGeometries, ui.listGeometries, ui.sliderOpacity });

	const CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	const auto visibleGeometries = m_viewSettings->VisibleGeometries();

	// visibility of geometries
	ui.groupBoxShowGeometries->setChecked(visibilitySettings.geometries);

	// geometries
	ui.listGeometries->clear();
	for (const auto& geometry : m_pSystemStructure->AllGeometries())
	{
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(geometry->Name()), ui.listGeometries);
		item->setData(Qt::UserRole, QString::fromStdString(geometry->Key()));
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(SetContains(visibleGeometries, geometry->Key()) ? Qt::Checked : Qt::Unchecked);
		ui.listGeometries->addItem(item);
	}

	// transparency
	ui.sliderOpacity->setSliderPosition(static_cast<int>((1 - m_viewSettings->GeometriesTransparency()) * 10));
	ui.labelTransparency->setText(QString::number(1 - m_viewSettings->GeometriesTransparency()));
}

void CViewOptionsTab::UpdateVolumesVisibility() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.groupBoxShowVolumes, ui.listVolumes });

	const CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	const auto visibleVolumes = m_viewSettings->VisibleVolumes();

	// visibility of volumes
	ui.groupBoxShowVolumes->setChecked(visibilitySettings.volumes);

	// volumes
	ui.listVolumes->clear();
	for (const auto& volume : m_pSystemStructure->AllAnalysisVolumes())
	{
		QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(volume->Name()), ui.listVolumes);
		item->setData(Qt::UserRole, QString::fromStdString(volume->Key()));
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
		item->setCheckState(SetContains(visibleVolumes, volume->Key()) ? Qt::Checked : Qt::Unchecked);
		ui.listVolumes->addItem(item);
	}
}

void CViewOptionsTab::UpdateVisibilityCheckboxes() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.checkBoxDomain, ui.checkBoxPBC });

	const CViewSettings::SVisibility visibilitySettings = m_viewSettings->Visibility();
	ui.checkBoxDomain->setChecked(visibilitySettings.domain);
	ui.checkBoxPBC->setChecked(visibilitySettings.pbc);
}

void CViewOptionsTab::SelectAllParticleMaterials() const
{
	auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();
	visiblePartMaterials = m_pSystemStructure->GetAllParticlesCompounds();
	m_viewSettings->VisiblePartMaterials(visiblePartMaterials);
}

void CViewOptionsTab::SelectAllBondMaterials() const
{
	auto visibleBondMaterials = m_viewSettings->VisibleBondMaterials();
	visibleBondMaterials = m_pSystemStructure->GetAllBondsCompounds();
	m_viewSettings->VisibleBondMaterials(visibleBondMaterials);
}

void CViewOptionsTab::SelectAllGeometries() const
{
	auto visibleGeometries = m_viewSettings->VisibleGeometries();
	visibleGeometries.clear();
	for (const auto& geometry : m_pSystemStructure->AllGeometries())
		visibleGeometries.insert(geometry->Key());
	m_viewSettings->VisibleGeometries(visibleGeometries);
}

void CViewOptionsTab::SelectAllVolumes() const
{
	auto visibleVolumes = m_viewSettings->VisibleVolumes();
	visibleVolumes.clear();
	for (const auto& volume : m_pSystemStructure->AllAnalysisVolumes())
		visibleVolumes.insert(volume->Key());
	m_viewSettings->VisibleVolumes(visibleVolumes);
}

void CViewOptionsTab::UpdateSelectedObjectsInfo() const
{
	ui.frameSelectingHint->setVisible(m_viewSettings->SelectedObjects().empty());
	ui.objectsInfoTable->setVisible(!m_viewSettings->SelectedObjects().empty());
	ui.mainToolBox->currentWidget()->update(); // to force apply visibility

	if (m_viewSettings->SelectedObjects().empty()) return;

	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::TIME, 0), m_dCurrentTime);

	if (m_viewSettings->SelectedObjects().size() == 1)
		UpdateOneObjectInfo();
	else
		UpdateGroupObjectsInfo();
}

void CViewOptionsTab::UpdateOneObjectInfo() const
{
	const size_t id = m_viewSettings->SelectedObjects().front();
	const CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(id);
	if (!object) return;

	ui.objectsInfoTable->item(EInfoRow::TYPE, 0)->setText("Particle");
	ui.objectsInfoTable->verticalHeaderItem(EInfoRow::ID)->setText("ID");
	ui.objectsInfoTable->item(EInfoRow::ID, 0)->setText(QString::number(id));

	if (const auto* part = dynamic_cast<const CSphere*>(object))
	{
		ShowConvValue(ui.objectsInfoTable->item(EInfoRow::RADIUS, 0), part->GetRadius(), EUnitType::PARTICLE_DIAMETER);
		ShowConvValue(ui.objectsInfoTable->item(EInfoRow::VOLUME, 0), PI * std::pow(2 * part->GetRadius(), 3.) / 6., EUnitType::VOLUME);
	}
	ShowVectorInTableColumn(object->GetCoordinates(m_dCurrentTime), ui.objectsInfoTable, EInfoRow::COORDX, 0, EUnitType::LENGTH);
	ShowVectorInTableColumn(object->GetVelocity(m_dCurrentTime), ui.objectsInfoTable, EInfoRow::VELOX, 0, EUnitType::VELOCITY);
	ShowVectorInTableColumn(object->GetAngleVelocity(m_dCurrentTime), ui.objectsInfoTable, EInfoRow::ANGVELOX, 0);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::MASS, 0), object->GetMass(), EUnitType::MASS);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::TEMPERATURE, 0), object->GetTemperature(m_dCurrentTime), EUnitType::TEMPERATURE);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::FORCE, 0), object->GetForce(m_dCurrentTime).Length(), EUnitType::FORCE);
	if (const CCompound* compound = m_pMaterialsDB->GetCompound(object->GetCompoundKey()))
		ui.objectsInfoTable->item(EInfoRow::MATERIAL, 0)->setText(ss2qs(compound->GetName()));
	else
		ui.objectsInfoTable->item(EInfoRow::MATERIAL, 0)->setText("Undefined");
}

void CViewOptionsTab::UpdateGroupObjectsInfo() const
{
	const auto objects = m_viewSettings->SelectedObjects();

	CVector3 coordCenter{ 0 }, velCenter{ 0 };
	double totalMass = 0, totalVolume = 0;
	QString material;

	for (auto id : objects)
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(id);
		if (!object) continue;
		double mass = object->GetMass();
		coordCenter += object->GetCoordinates(m_dCurrentTime) * mass;
		velCenter += object->GetVelocity(m_dCurrentTime) * mass;
		totalMass += mass;
		if (const auto* part = dynamic_cast<const CSphere*>(object))
			totalVolume += PI * std::pow(2 * part->GetRadius(), 3.) / 6.;

		QString currMaterial = "Undefined";
		if (const CCompound* compound = m_pMaterialsDB->GetCompound(object->GetCompoundKey()))
			currMaterial = ss2qs(compound->GetName());

		if (material.isEmpty())
			material = currMaterial;
		else if (currMaterial != material && material != "Undefined")
			material = "<Mixed>";
	}

	coordCenter = coordCenter / totalMass;
	velCenter = velCenter / totalMass;

	ui.objectsInfoTable->item(EInfoRow::TYPE, 0)->setText("Agglomerate");
	ui.objectsInfoTable->verticalHeaderItem(EInfoRow::ID)->setText("Count");
	ui.objectsInfoTable->item(EInfoRow::ID, 0)->setText(QString::number(objects.size()));
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::RADIUS, 0), 0, EUnitType::PARTICLE_DIAMETER);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::VOLUME, 0), totalVolume, EUnitType::VOLUME);
	ShowVectorInTableColumn(coordCenter, ui.objectsInfoTable, EInfoRow::COORDX, 0, EUnitType::LENGTH);
	ShowVectorInTableColumn(velCenter, ui.objectsInfoTable, EInfoRow::VELOX, 0, EUnitType::VELOCITY);
	ShowVectorInTableColumn(CVector3{ 0 }, ui.objectsInfoTable, EInfoRow::ANGVELOX, 0);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::MASS, 0), totalMass, EUnitType::MASS);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::TEMPERATURE, 0), 0, EUnitType::TEMPERATURE);
	ShowConvValue(ui.objectsInfoTable->item(EInfoRow::FORCE, 0), 0, EUnitType::FORCE);
	ui.objectsInfoTable->item(EInfoRow::MATERIAL, 0)->setText(material);
}

void CViewOptionsTab::UpdateColoringTypes()
{
	// function to add radio button into group box
	const auto InsertItem = [&](EColoring _type, EUnitType _units, const QString& _text, const QString& _tooltip)
	{
		auto* radio = new QRadioButton(_text, this);
		radio->setProperty("units", E2I(_units));         // add property to store information about measurement units
		ui.groupBoxColorType->layout()->addWidget(radio); // put radio button on layout
		m_coloringGroup.addButton(radio, E2I(_type));     // add radio button to button group for simpler management
		// add help text
		radio->setToolTip(_tooltip);
		radio->setStatusTip(_tooltip);
		radio->setWhatsThis(_tooltip);
	};

	/*[[maybe_unused]]*/ QSignalBlocker blocker(m_coloringGroup);
	for (auto w : ui.groupBoxColorType->findChildren<QWidget*>(QString{}, Qt::FindDirectChildrenOnly)) delete w; // remove all radio buttons

	InsertItem(EColoring::NONE,               EUnitType::NONE,              "None",					"No coloring");
	if (m_pSystemStructure->IsSolidBondsExist())
	{
	InsertItem(EColoring::AGGL_SIZE,          EUnitType::NONE,              "Agglomerate size",		"Coloring by number of particles in agglomerates");
	InsertItem(EColoring::BOND_TOTAL_FORCE,   EUnitType::FORCE,             "Bond force",			"Coloring by forces in bonds");
	InsertItem(EColoring::BOND_NORMAL_STRESS, EUnitType::STRESS,            "Bond norm stress",		"Coloring by normal stress in bonds");
	InsertItem(EColoring::BOND_STRAIN,        EUnitType::NONE,              "Bond strain",			"Coloring by strain in bonds");
	}
	if (m_pSystemStructure->IsContactRadiusEnabled())
	{
	InsertItem(EColoring::CONTACT_DIAMETER,   EUnitType::PARTICLE_DIAMETER, "Contact diameter",		"Coloring by contact diameters");
	}
	InsertItem(EColoring::COORDINATE,         EUnitType::LENGTH,            "Coordinate",			"Coloring by coordinates");
	InsertItem(EColoring::COORD_NUMBER,       EUnitType::NONE,              "Coordination number",	"Coloring by coordination numbers");
	InsertItem(EColoring::DIAMETER,           EUnitType::PARTICLE_DIAMETER, "Diameter",				"Coloring by diameters of particles and bonds");
	InsertItem(EColoring::DISPLACEMENT,       EUnitType::LENGTH,			"Displacement",			"Coloring by displacement of particles and bonds");
	InsertItem(EColoring::FORCE,              EUnitType::FORCE,             "Force",				"Coloring by forces");
	InsertItem(EColoring::MATERIAL,           EUnitType::NONE,              "Material",				"Coloring by materials");
	InsertItem(EColoring::OVERLAP,            EUnitType::LENGTH,            "Maximum overlap",		"Coloring by overlaps");
	InsertItem(EColoring::ANGLE_VELOCITY,     EUnitType::ANGULAR_VELOCITY,  "Rotation velocity",	"Coloring by rotation velocities");
	InsertItem(EColoring::STRESS,             EUnitType::STRESS,            "Stress",				"Coloring by stresses");
	InsertItem(EColoring::PRINCIPAL_STRESS, EUnitType::STRESS, "Principal stress", "Coloring by principal stresses");
	InsertItem(EColoring::VELOCITY,           EUnitType::VELOCITY,          "Velocity",				"Coloring by velocities");
	InsertItem(EColoring::TEMPERATURE,		  EUnitType::TEMPERATURE,       "Temperature",			"Coloring by temperature");

	m_coloringGroup.button(E2I(m_viewSettings->Coloring().type))->setChecked(true); // select property
}

void CViewOptionsTab::UpdateColoringLimits() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.lineEditColorLimitMin, ui.lineEditColorLimitMax });
	// get coloring type
	const EColoring type = m_viewSettings->Coloring().type;
	// get measurement units from property of the corresponding radio button
	const EUnitType units = static_cast<EUnitType>(m_coloringGroup.button(E2I(type))->property("units").toUInt());
	// convert values from SI to selected
	const double min = m_pUnitConverter->GetValue(units, m_viewSettings->Coloring().minValue);
	const double max = m_pUnitConverter->GetValue(units, m_viewSettings->Coloring().maxValue);
	// set values
	ui.lineEditColorLimitMin->setText(QString::number(min));
	ui.lineEditColorLimitMax->setText(QString::number(max));
	// change labels according to selected units
	ShowConvLabel(ui.labelColorLimitMin, "Min", units);
	ShowConvLabel(ui.labelColorLimitMax, "Max", units);
}

void CViewOptionsTab::UpdateColoringComponent() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.radioButtonComponentL, ui.radioButtonComponentX, ui.radioButtonComponentY, ui.radioButtonComponentZ });
	// select proper component
	m_componentGroup.button(E2I(m_viewSettings->Coloring().component))->setChecked(true);
	// change activity of components
	ui.groupBoxComponent->setEnabled(VectorContains(m_coloringVectors, m_viewSettings->Coloring().type));
}

void CViewOptionsTab::UpdateColoringColors() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker({ ui.colorMin, ui.colorMid, ui.colorMax });
	ui.colorMin->SetColor(m_viewSettings->Coloring().minColor);
	ui.colorMid->SetColor(m_viewSettings->Coloring().midColor);
	ui.colorMax->SetColor(m_viewSettings->Coloring().maxColor);
}

void CViewOptionsTab::UpdateLabels() const
{
	// cutting
	ShowConvLabel(ui.xMinLabel,        "Min",         EUnitType::LENGTH);
	ShowConvLabel(ui.xMaxLabel,        "Max",         EUnitType::LENGTH);
	ShowConvLabel(ui.labelSliceCoords, "Coordinates", EUnitType::LENGTH);

	// visibility
	ShowConvLabel(ui.labelTimeBegBrokenBonds, "From", EUnitType::TIME);
	ShowConvLabel(ui.labelTimeEndBrokenBonds, "To",   EUnitType::TIME);

	// selection
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::TIME),			"Time",			EUnitType::TIME);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::RADIUS),		"Radius",		EUnitType::PARTICLE_DIAMETER);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::VOLUME),		"Volume",		EUnitType::VOLUME);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::COORDX),		"X",			EUnitType::LENGTH);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::COORDY),		"Y",			EUnitType::LENGTH);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::COORDZ),		"Z",			EUnitType::LENGTH);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::VELOX),			"Vx",			EUnitType::VELOCITY);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::VELOY),			"Vy",			EUnitType::VELOCITY);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::VELOZ),			"Vz",			EUnitType::VELOCITY);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::ANGVELOX),		"Wx",			EUnitType::ANGULAR_VELOCITY);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::ANGVELOY),		"Wy",			EUnitType::ANGULAR_VELOCITY);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::ANGVELOZ),		"Wz",			EUnitType::ANGULAR_VELOCITY);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::MASS),			"Mass",			EUnitType::MASS);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::TEMPERATURE),	"Temperature",	EUnitType::TEMPERATURE);
	ShowConvLabel(ui.objectsInfoTable->verticalHeaderItem(EInfoRow::FORCE),			"Force",		EUnitType::FORCE);
}
