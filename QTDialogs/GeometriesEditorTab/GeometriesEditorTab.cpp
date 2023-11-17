/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesEditorTab.h"
#include "MeshGenerator.h"
#include "QtSignalBlocker.h"
#include <QMenu>
#include <QWidgetAction>
#include <QMessageBox>
#include <QToolButton>

CGeometriesEditorTab::CGeometriesEditorTab(QWidget* parent) : CMusenDialog{ parent }
{
	ui.setupUi(this);
	setWindowFlags(Qt::Window);

	SetupGeometriesList();
	SetupPropertiesList();

	InitializeConnections();

	ui.splitter->setSizes(QList<int>{100, 160, 150});
	m_motionWidth = ui.groupMotion->width();
	ui.tableMotion->BlockingPaste(true);

	m_sHelpFileName = "Users Guide/Geometries Editor.pdf";
}

void CGeometriesEditorTab::InitializeConnections() const
{
	connect(ui.buttonDeleteGeometry, &QPushButton::clicked, this, &CGeometriesEditorTab::DeleteGeometry);
	connect(ui.buttonUpGeometry,	 &QPushButton::clicked, this, &CGeometriesEditorTab::UpGeometry);
	connect(ui.buttonDownGeometry,	 &QPushButton::clicked, this, &CGeometriesEditorTab::DownGeometry);

	connect(ui.listGeometries, &QTreeWidget::itemChanged,		 this, &CGeometriesEditorTab::NameChanged);
	connect(ui.listGeometries, &QTreeWidget::currentItemChanged, this, &CGeometriesEditorTab::GeometrySelected);

	connect(ui.tableMotion,				&CQtTable::itemChanged,				this, &CGeometriesEditorTab::MotionTableChanged);
	connect(ui.tableMotion,				&CQtTable::ComboBoxIndexChanged,	this, &CGeometriesEditorTab::MotionTableChanged);
	connect(ui.tableMotion,				&CQtTable::DataPasted,				this, &CGeometriesEditorTab::MotionTableChanged);
	connect(ui.checkBoxAroundCenter,	&QCheckBox::clicked,				this, &CGeometriesEditorTab::MotionChanged);
	connect(ui.checkBoxFreeMotionX,		&QCheckBox::clicked,				this, &CGeometriesEditorTab::MotionChanged);
	connect(ui.checkBoxFreeMotionY,		&QCheckBox::clicked,				this, &CGeometriesEditorTab::MotionChanged);
	connect(ui.checkBoxFreeMotionZ,		&QCheckBox::clicked,				this, &CGeometriesEditorTab::MotionChanged);
	connect(ui.lineEditMass,			&QLineEdit::editingFinished,		this, &CGeometriesEditorTab::MotionChanged);

	connect(ui.buttonAddMotion,		&QPushButton::clicked, this, &CGeometriesEditorTab::AddMotion);
	connect(ui.buttonDeleteMotion,	&QPushButton::clicked, this, &CGeometriesEditorTab::DeleteMotion);
}

void CGeometriesEditorTab::Initialize()
{
	ui.tableMotion->SetUnitConverter(m_pUnitConverter);
	ui.treeProperties->SetUnitConverter(m_pUnitConverter);
}

void CGeometriesEditorTab::SetupGeometriesList()
{
	auto font = ui.listGeometries->font();
	font.setItalic(true);

	ui.listGeometries->clear();
	m_list[EType::GEOMETRY] = ui.listGeometries->CreateItem(0, "Real geometries", CQtTree::EFlags::NoEdit | CQtTree::EFlags::NoSelect);
	m_list[EType::GEOMETRY]->setFont(0, font);

	m_list[EType::VOLUME] = ui.listGeometries->CreateItem(0, "Analysis volumes", CQtTree::EFlags::NoEdit | CQtTree::EFlags::NoSelect);
	m_list[EType::VOLUME]->setFont(0, font);
}

void CGeometriesEditorTab::SetupPropertiesList()
{
	/* Most of names are set to elements only for proper automatic resizing of items.
	 * They will be replaced with proper names in update functions.*/

	ui.treeProperties->setColumnCount(3);

	/// general
	auto* general = ui.treeProperties->CreateItem(0, "General");

	// combo box with materials
	m_properties[EProperty::MATERIAL] = ui.treeProperties->CreateItem(general, 0, "Material");
	const auto* material = ui.treeProperties->AddComboBox(m_properties[EProperty::MATERIAL], 1, {}, {}, 0);
	connect(material, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &CGeometriesEditorTab::MaterialChanged);

	// color picker
	m_properties[EProperty::COLOR] = ui.treeProperties->CreateItem(general, 0, "Color");
	const auto* color = ui.treeProperties->AddColorPicker(m_properties[EProperty::COLOR], 1, {});
	connect(color, &CColorView::ColorEdited, this, &CGeometriesEditorTab::ColorChanged);

	// motion
	m_properties[EProperty::MOTION] = ui.treeProperties->CreateItem(general, 0, "Motion");
	const auto* motion = ui.treeProperties->AddComboBox(m_properties[EProperty::MOTION], 1, 
		{ "None", "Time-dependent", "Force-dependent", "Constant force" }, 
		{ E2I(CGeometryMotion::EMotionType::NONE), E2I(CGeometryMotion::EMotionType::TIME_DEPENDENT),
		E2I(CGeometryMotion::EMotionType::FORCE_DEPENDENT), E2I(CGeometryMotion::EMotionType::CONSTANT_FORCE) }, 0);
	connect(motion, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &CGeometriesEditorTab::MotionTypeChanged);

	// triangles
	m_properties[EProperty::TRIANGLES] = ui.treeProperties->CreateItem(general, 0, "Triangles");
	const auto* triangles = ui.treeProperties->AddListSpinBox(m_properties[EProperty::TRIANGLES], 1, {}, 0);
	connect(triangles, static_cast<void (CQtListSpinBox::*)(int)>(&CQtListSpinBox::valueChanged), this, &CGeometriesEditorTab::QualityChanged);

	general->setExpanded(true);

	/// rotation
	auto* rotation = ui.treeProperties->CreateItem(0, "Rotation");

	auto AddRotation = [&](EProperty _prop, const std::string& _name, EAxis _axis)
	{
		m_properties[_prop] = ui.treeProperties->CreateItem(rotation, 0, _name);
		ui.treeProperties->AddDoubleSpinBox(m_properties[_prop], 1, {}, EUnitType::ANGLE);
		const auto* rotate = ui.treeProperties->AddPushButton(m_properties[_prop], 2, "Apply");
		connect(rotate, &QPushButton::clicked, [=] { RotationChanged(_axis); });
	};
	AddRotation(EProperty::ROTATE_X, "X [rad]", EAxis::X);
	AddRotation(EProperty::ROTATE_Y, "Y [rad]", EAxis::Y);
	AddRotation(EProperty::ROTATE_Z, "Z [rad]", EAxis::Z);

	rotation->setExpanded(true);

	/// position
	auto* position = ui.treeProperties->CreateItem(0, "Position");

	auto AddPosition = [&](EProperty _prop, const std::string& _name, const QString& _buttonName1, const QString& _buttonName2, EAxis _axis)
	{
		m_properties[_prop] = ui.treeProperties->CreateItem(position, 0, _name);
		const auto* spinbox = ui.treeProperties->AddDoubleSpinBox(m_properties[_prop], 1, {}, EUnitType::LENGTH);
		connect(spinbox, &CQtDoubleSpinBox::ValueChanged, this, &CGeometriesEditorTab::PositionChanged);

		auto* widget = new QWidget{ ui.treeProperties };
		auto* button1 = new QPushButton{ widget };
		auto* button2 = new QPushButton{ widget };
		button1->setIcon(QIcon{ ":/MusenGUI/Pictures/geo_" + _buttonName1 + ".png" });
		button2->setIcon(QIcon{ ":/MusenGUI/Pictures/geo_" + _buttonName2 + ".png" });
		button1->setAutoDefault(false);
		button2->setAutoDefault(false);
		button1->setToolTip("Place geometry on the selected side of all particles");
		button2->setToolTip("Place geometry on the selected side of all particles");
		connect(button1, &QPushButton::clicked, this, [=]() { PlaceAside(_axis, EPosition::MIN); });
		connect(button2, &QPushButton::clicked, this, [=]() { PlaceAside(_axis, EPosition::MAX); });
		auto* layout = new QHBoxLayout{ widget };
		layout->setAlignment(Qt::AlignCenter);
		layout->setContentsMargins(0, 0, 0, 0);
		layout->addWidget(button1);
		layout->addWidget(button2);
		widget->setLayout(layout);
		ui.treeProperties->setItemWidget(m_properties[_prop], 2, widget);
	};
	AddPosition(EProperty::POSITION_X, "X [mm]", "left", "right", EAxis::X);
	AddPosition(EProperty::POSITION_Y, "Y [mm]", "front", "back", EAxis::Y);
	AddPosition(EProperty::POSITION_Z, "Z [mm]", "bottom", "top", EAxis::Z);

	position->setExpanded(true);

	/// size
	auto* sizes = ui.treeProperties->CreateItem(0, "Size");

	// scaling
	m_properties[EProperty::SCALE] = ui.treeProperties->CreateItem(sizes, 0, "Scale");
	auto* scale = ui.treeProperties->AddDoubleSpinBox(m_properties[EProperty::SCALE], 1, {}, EUnitType::NONE);
	scale->AllowOnlyPositive(true);
	connect(scale, &CQtDoubleSpinBox::ValueChanged, this, &CGeometriesEditorTab::ScalingChanged);

	// sizes
	auto AddSize = [&](EProperty _prop, const std::string& _name)
	{
		m_properties[_prop] = ui.treeProperties->CreateItem(sizes, 0, _name);
		auto* spinbox = ui.treeProperties->AddDoubleSpinBox(m_properties[_prop], 1, {}, EUnitType::LENGTH);
		spinbox->AllowOnlyPositive(true);
		connect(spinbox, &CQtDoubleSpinBox::ValueChanged, this, &CGeometriesEditorTab::SizeChanged);
	};
	AddSize(EProperty::SIZE_X,       "X [mm]");
	AddSize(EProperty::SIZE_Y,       "Y [mm]");
	AddSize(EProperty::SIZE_Z,       "Z [mm]");
	AddSize(EProperty::WIDTH,        "Width [mm]");
	AddSize(EProperty::DEPTH,        "Depth [mm]");
	AddSize(EProperty::HEIGHT,       "Height [mm]");
	AddSize(EProperty::RADIUS,       "Radius [mm]");
	AddSize(EProperty::INNER_RADIUS, "Inner radius [mm]");

	sizes->setExpanded(true);

	/// resize columns and contract some rows
	for (int i = 0; i < ui.treeProperties->columnCount(); ++i)
		ui.treeProperties->resizeColumnToContents(i);
	rotation->setExpanded(false);
}

void CGeometriesEditorTab::setVisible(bool _visible)
{
	CMusenDialog::setVisible(_visible);
	if (!_visible) return;
	if (m_object) return; // geometry is already selected
	if (m_list[EType::GEOMETRY]->childCount() != 0)		// select first available geometry
		ui.listGeometries->setCurrentItem(m_list[EType::GEOMETRY]->child(0));
	else if (m_list[EType::VOLUME]->childCount() != 0)	// select first available volume
		ui.listGeometries->setCurrentItem(m_list[EType::VOLUME]->child(0));
}

void CGeometriesEditorTab::UpdateWholeView()
{
	UpdateMeasurementUnits();
	UpdateAddButtons();
	UpdateMaterialsCombo();
	UpdateGeometriesList();
	UpdatePropertiesInfo();
	UpdateMotionInfo();
}

void CGeometriesEditorTab::UpdateMeasurementUnits() const
{
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::ROTATE_X),     "X",            EUnitType::ANGLE);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::ROTATE_Y),     "Y",            EUnitType::ANGLE);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::ROTATE_Z),     "Z",            EUnitType::ANGLE);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::POSITION_X),   "X",            EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::POSITION_Y),   "Y",            EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::POSITION_Z),   "Z",            EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::SIZE_X),       "X",            EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::SIZE_Y),       "Y",            EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::SIZE_Z),       "Z",            EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::WIDTH),        "Width",        EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::DEPTH),        "Depth",        EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::HEIGHT),       "Height",       EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::RADIUS),       "Radius",       EUnitType::LENGTH);
	ui.treeProperties->SetHeaderText(m_properties.at(EProperty::INNER_RADIUS), "Inner radius", EUnitType::LENGTH);

	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::TIME_BEG,		"Time start",		EUnitType::TIME);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::TIME_END,		"Time end",			EUnitType::TIME);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::FORCE,			"Force Z",			EUnitType::FORCE);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::VEL_X,			"Velocity X",		EUnitType::VELOCITY);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::VEL_Y,			"Velocity Y",		EUnitType::VELOCITY);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::VEL_Z,			"Velocity Z",		EUnitType::VELOCITY);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::ROT_VEL_X,		"Rot. velocity X",  EUnitType::ANGULAR_VELOCITY);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::ROT_VEL_Y,		"Rot. velocity Y",  EUnitType::ANGULAR_VELOCITY);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::ROT_VEL_Z,		"Rot. velocity Z",  EUnitType::ANGULAR_VELOCITY);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::ROT_CENTER_X,	"Rot. center X",	EUnitType::LENGTH);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::ROT_CENTER_Y,	"Rot. center Y",	EUnitType::LENGTH);
	ui.tableMotion->SetRowHeaderItemConv(ERowMotion::ROT_CENTER_Z,	"Rot. center Z",	EUnitType::LENGTH);

	ShowConvLabel(ui.labelMass, "Mass", EUnitType::MASS);
}

void CGeometriesEditorTab::UpdateAddButtons()
{
	// function to add a text separator to menu
	auto AddSeparator = [&](QMenu* _menu, const QString& _text)
	{
		auto* label = new QLabel{ "<b>" + _text + "</b>", this };
		label->setAlignment(Qt::AlignCenter);
		auto* action = new QWidgetAction{ _menu };
		action->setDefaultWidget(label);
		_menu->addAction(action);
	};

	// function to create a menu for push button
	auto CreateMenu = [&](EType _type)
	{
		auto* geometriesMenu = new QMenu(this);

		AddSeparator(geometriesMenu, "Standard");

		for (const auto& v : AllStandardVolumeTypes())
		{
			if (v.first == EVolumeShape::VOLUME_STL) continue;
			auto* action = new QAction{ QString::fromStdString(v.second), this };
			geometriesMenu->addAction(action);
			connect(action, &QAction::triggered, this, [=]() { AddGeometryStd(v.first, _type); });
		}

		AddSeparator(geometriesMenu, "From geometries DB");

		for (const auto& g : m_pGeometriesDB->Geometries())
		{
			auto* action = new QAction{ QString::fromStdString(g->mesh.Name()), this };
			geometriesMenu->addAction(action);
			connect(action, &QAction::triggered, this, [=]() { AddGeometryLib(g->key, _type); });
		}

		return geometriesMenu;
	};

	ui.buttonAddGeometry->setMenu(CreateMenu(EType::GEOMETRY));
	ui.buttonAddVolume->setMenu(CreateMenu(EType::VOLUME));
}

void CGeometriesEditorTab::UpdateMaterialsCombo() const
{
	ui.treeProperties->SetupComboBox(m_properties.at(EProperty::MATERIAL), 1, m_pMaterialsDB->GetCompoundsNames(), m_pMaterialsDB->GetCompoundsKeys(), "");
}

void CGeometriesEditorTab::UpdateMotionCombo() const
{
	switch(Type())
	{
	case EType::NONE: break;
	case EType::GEOMETRY:
		ui.treeProperties->SetupComboBox(m_properties.at(EProperty::MOTION), 1, { "None", "Time-dependent", "Force-dependent", "Constant force" }, 
			{ E2I(CGeometryMotion::EMotionType::NONE), E2I(CGeometryMotion::EMotionType::TIME_DEPENDENT), 
			E2I(CGeometryMotion::EMotionType::FORCE_DEPENDENT), E2I(CGeometryMotion::EMotionType::CONSTANT_FORCE) }, -1);
		break;
	case EType::VOLUME:
		ui.treeProperties->SetupComboBox(m_properties.at(EProperty::MOTION), 1, { "None", "Time-dependent" }, { E2I(CGeometryMotion::EMotionType::NONE), E2I(CGeometryMotion::EMotionType::TIME_DEPENDENT) }, -1);
		break;
	}
}

void CGeometriesEditorTab::UpdateGeometriesList()
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker{ ui.listGeometries };
	const auto oldKey = ui.listGeometries->GetCurrentData();

	CQtTree::Clear(m_list[EType::GEOMETRY]);
	m_list[EType::GEOMETRY]->setHidden(m_pSystemStructure->GeometriesNumber() == 0);
	for (const auto& geometry : m_pSystemStructure->AllGeometries())
		ui.listGeometries->CreateItem(m_list[EType::GEOMETRY], 0, geometry->Name(), CQtTree::EFlags::Edit, QString::fromStdString(geometry->Key()));

	CQtTree::Clear(m_list[EType::VOLUME]);
	m_list[EType::VOLUME]->setHidden(m_pSystemStructure->AnalysisVolumesNumber() == 0);
	for (const auto& volume : m_pSystemStructure->AllAnalysisVolumes())
		ui.listGeometries->CreateItem(m_list[EType::VOLUME], 0, volume->Name(), CQtTree::EFlags::Edit, QString::fromStdString(volume->Key()));

	ui.listGeometries->expandAll();
	ui.listGeometries->SetCurrentItem(oldKey);
	GeometrySelected();
}

void CGeometriesEditorTab::UpdatePropertiesInfoEmpty() const
{
	/*[[maybe_unused]]*/ CQtSignalBlocker blocker{ ui.treeProperties };

	ui.treeProperties->SetComboBoxValue(     m_properties.at(EProperty::MATERIAL),     1, -1);
	ui.treeProperties->SetComboBoxValue(     m_properties.at(EProperty::MOTION),       1, 0);
	ui.treeProperties->SetListSpinBoxValue(  m_properties.at(EProperty::TRIANGLES),    1, 0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::ROTATE_X),     1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::ROTATE_Y),     1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::ROTATE_Z),     1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_X),   1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_Y),   1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_Z),   1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SCALE),        1, 1.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_X),       1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_Y),       1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_Z),       1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::WIDTH),        1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::DEPTH),        1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::HEIGHT),       1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::RADIUS),       1, 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::INNER_RADIUS), 1, 0.0);
	m_properties.at(EProperty::SIZE_X      )->setHidden(true);
	m_properties.at(EProperty::SIZE_Y      )->setHidden(true);
	m_properties.at(EProperty::SIZE_Z      )->setHidden(true);
	m_properties.at(EProperty::WIDTH       )->setHidden(true);
	m_properties.at(EProperty::DEPTH       )->setHidden(true);
	m_properties.at(EProperty::HEIGHT      )->setHidden(true);
	m_properties.at(EProperty::RADIUS      )->setHidden(true);
	m_properties.at(EProperty::INNER_RADIUS)->setHidden(true);
}

void CGeometriesEditorTab::UpdatePropertiesInfo() const
{
	ui.treeProperties->setEnabled(m_object);

	if (!m_object)
	{
		UpdatePropertiesInfoEmpty();
		return;
	}

	CQtSignalBlocker blocker{ ui.treeProperties };

	ui.treeProperties->itemWidget(m_properties.at(EProperty::MATERIAL), 1)->setEnabled(Type() == EType::GEOMETRY);
	ui.treeProperties->SetComboBoxValue(m_properties.at(EProperty::MATERIAL), 1, Type() == EType::GEOMETRY ? QString::fromStdString(dynamic_cast<CRealGeometry*>(m_object)->Material()) : "");
	ui.treeProperties->SetColorPickerValue(m_properties.at(EProperty::COLOR), 1, m_object->Color());
	ui.treeProperties->SetComboBoxValue(m_properties.at(EProperty::MOTION), 1, E2I(m_object->Motion()->MotionType()));
	ui.treeProperties->itemWidget(m_properties.at(EProperty::TRIANGLES), 1)->setEnabled(m_object->Shape() != EVolumeShape::VOLUME_STL && m_object->Shape() != EVolumeShape::VOLUME_BOX);
	ui.treeProperties->SetupListSpinBox(m_properties.at(EProperty::TRIANGLES), 1, vector_cast<int>(CMeshGenerator::AllowedTrianglesNumber(m_object->Shape())), static_cast<int>(m_object->TrianglesNumber()));
	const auto center = m_object->Center();
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_X), 1, std::abs(center.x) > 1e-12 ? center.x : 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_Y), 1, std::abs(center.y) > 1e-12 ? center.y : 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_Z), 1, std::abs(center.z) > 1e-12 ? center.z : 0.0);
	ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SCALE),      1, m_object->ScalingFactor());
	const auto shape = m_object->Shape();
	if (shape == EVolumeShape::VOLUME_STL)
	{
		const auto bb = m_object->BoundingBox();
		const auto size = bb.coordEnd - bb.coordBeg;
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_X), 1, size.x);
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_Y), 1, size.y);
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_Z), 1, size.z);
	}
	else
	{
		const auto sizes = m_object->Sizes();
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::WIDTH),        1, sizes.Width());
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::DEPTH),        1, sizes.Depth());
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::HEIGHT),       1, sizes.Height());
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::RADIUS),       1, sizes.Radius());
		ui.treeProperties->SetDoubleSpinBoxValue(m_properties.at(EProperty::INNER_RADIUS), 1, sizes.InnerRadius());
	}

	m_properties.at(EProperty::SIZE_X      )->setHidden(!VectorContains({ EVolumeShape::VOLUME_STL }, shape));
	m_properties.at(EProperty::SIZE_Y      )->setHidden(!VectorContains({ EVolumeShape::VOLUME_STL }, shape));
	m_properties.at(EProperty::SIZE_Z      )->setHidden(!VectorContains({ EVolumeShape::VOLUME_STL }, shape));
	m_properties.at(EProperty::WIDTH       )->setHidden(!VectorContains({ EVolumeShape::VOLUME_BOX }, shape));
	m_properties.at(EProperty::DEPTH       )->setHidden(!VectorContains({ EVolumeShape::VOLUME_BOX }, shape));
	m_properties.at(EProperty::HEIGHT      )->setHidden(!VectorContains({ EVolumeShape::VOLUME_BOX, EVolumeShape::VOLUME_CYLINDER }, shape));
	m_properties.at(EProperty::RADIUS      )->setHidden(!VectorContains({ EVolumeShape::VOLUME_CYLINDER, EVolumeShape::VOLUME_SPHERE, EVolumeShape::VOLUME_HOLLOW_SPHERE }, shape));
	m_properties.at(EProperty::INNER_RADIUS)->setHidden(!VectorContains({ EVolumeShape::VOLUME_HOLLOW_SPHERE }, shape));
}

void CGeometriesEditorTab::UpdateMotionInfo()
{
	ui.tableMotion->setColumnCount(0);
	ui.groupMotion->setEnabled(m_object);
	UpdateMotionVisibility();

	if (!m_object) return;

	CQtSignalBlocker blocker{ ui.tableMotion, ui.checkBoxAroundCenter, ui.groupFreeMotion, ui.checkBoxFreeMotionX, ui.checkBoxFreeMotionY, ui.checkBoxFreeMotionZ, ui.lineEditMass };

	const auto* geometry = dynamic_cast<CRealGeometry*>(m_object);

	// movement type
	const auto* motion = m_object->Motion();
	const auto type = motion->MotionType();
	ui.tableMotion->ShowRow(ERowMotion::TIME_BEG,     type == CGeometryMotion::EMotionType::TIME_DEPENDENT);
	ui.tableMotion->ShowRow(ERowMotion::TIME_END,     type == CGeometryMotion::EMotionType::TIME_DEPENDENT);
	ui.tableMotion->ShowRow(ERowMotion::FORCE,        type == CGeometryMotion::EMotionType::FORCE_DEPENDENT || type == CGeometryMotion::EMotionType::CONSTANT_FORCE);
	ui.tableMotion->ShowRow(ERowMotion::LIMIT_TYPE,   type == CGeometryMotion::EMotionType::FORCE_DEPENDENT || type == CGeometryMotion::EMotionType::CONSTANT_FORCE);
	ui.tableMotion->ShowRow(ERowMotion::ROT_VEL_X,    geometry);
	ui.tableMotion->ShowRow(ERowMotion::ROT_VEL_Y,    geometry);
	ui.tableMotion->ShowRow(ERowMotion::ROT_VEL_Z,    geometry);
	ui.tableMotion->ShowRow(ERowMotion::ROT_CENTER_X, geometry && !geometry->RotateAroundCenter());
	ui.tableMotion->ShowRow(ERowMotion::ROT_CENTER_Y, geometry && !geometry->RotateAroundCenter());
	ui.tableMotion->ShowRow(ERowMotion::ROT_CENTER_Z, geometry && !geometry->RotateAroundCenter());

	// movement parameters
	switch (type)
	{
	case CGeometryMotion::EMotionType::TIME_DEPENDENT:
	{
		const auto intervals = motion->GetTimeIntervals();
		ui.tableMotion->setColumnCount(static_cast<int>(intervals.size()));
		for (int i = 0; i < static_cast<int>(intervals.size()); ++i)
		{
			ui.tableMotion->SetItemEditableConv(    ERowMotion::TIME_BEG,     i, intervals[i].timeBeg,                 EUnitType::TIME);
			ui.tableMotion->SetItemEditableConv(    ERowMotion::TIME_END,     i, intervals[i].timeEnd,                 EUnitType::TIME);
			ui.tableMotion->SetItemsColEditableConv(ERowMotion::VEL_X,        i, intervals[i].motion.velocity,         EUnitType::VELOCITY);
			ui.tableMotion->SetItemsColEditableConv(ERowMotion::ROT_VEL_X,    i, intervals[i].motion.rotationVelocity, EUnitType::ANGULAR_VELOCITY);
			ui.tableMotion->SetItemsColEditableConv(ERowMotion::ROT_CENTER_X, i, intervals[i].motion.rotationCenter,   EUnitType::LENGTH);
		}
		break;
	}
	case CGeometryMotion::EMotionType::FORCE_DEPENDENT:
	case CGeometryMotion::EMotionType::CONSTANT_FORCE:
	{
		const auto intervals = motion->GetForceIntervals();
		ui.tableMotion->setColumnCount(static_cast<int>(intervals.size()));
		for (int i = 0; i < static_cast<int>(intervals.size()); ++i)
		{
			ui.tableMotion->SetItemEditableConv(    ERowMotion::FORCE,        i, intervals[i].forceLimit,              EUnitType::FORCE);
			ui.tableMotion->SetComboBox(            ERowMotion::LIMIT_TYPE,   i, { "MIN", "MAX" }, { E2I(CGeometryMotion::SForceMotionInterval::ELimitType::MIN), E2I(CGeometryMotion::SForceMotionInterval::ELimitType::MAX) }, E2I(intervals[i].limitType));
			ui.tableMotion->SetItemsColEditableConv(ERowMotion::VEL_X,        i, intervals[i].motion.velocity,         EUnitType::VELOCITY);
			ui.tableMotion->SetItemsColEditableConv(ERowMotion::ROT_VEL_X,    i, intervals[i].motion.rotationVelocity, EUnitType::ANGULAR_VELOCITY);
			ui.tableMotion->SetItemsColEditableConv(ERowMotion::ROT_CENTER_X, i, intervals[i].motion.rotationCenter,   EUnitType::LENGTH);
			if (type == CGeometryMotion::EMotionType::CONSTANT_FORCE) break; // show only one entry in constant-force mode
		}
		break;
	}
	case CGeometryMotion::EMotionType::NONE:
		ui.tableMotion->setColumnCount(0);
		break;
	}

	ui.checkBoxAroundCenter->setChecked(geometry && geometry->RotateAroundCenter());
	ui.checkBoxAroundCenter->setEnabled(geometry);
	ui.tableMotion->resizeColumnsToContents();

	// free motion parameters
	ui.checkBoxFreeMotionX->setChecked(geometry && geometry->FreeMotion().x);
	ui.checkBoxFreeMotionY->setChecked(geometry && geometry->FreeMotion().y);
	ui.checkBoxFreeMotionZ->setChecked(geometry && geometry->FreeMotion().z);
	ShowConvValue(ui.lineEditMass, geometry ? geometry->Mass() : 0, EUnitType::MASS);
	ui.lineEditMass->setEnabled(geometry && !geometry->FreeMotion().IsZero());
	ui.groupFreeMotion->setEnabled(geometry);
}

void CGeometriesEditorTab::UpdateMotionVisibility()
{
	if (Type() == EType::NONE || m_object->Motion()->MotionType() == CGeometryMotion::EMotionType::NONE)
	{
		if (ui.groupMotion->isVisible())
		{
			m_motionWidth = ui.groupMotion->width();
			ui.groupMotion->hide();
			resize(width() - m_motionWidth - ui.groupMotion->layout()->spacing() + 1, height());
		}
	}
	else
	{
		if (ui.groupMotion->isHidden())
		{
			ui.groupMotion->show();
			resize(width() + m_motionWidth + ui.groupMotion->layout()->spacing() - 1, height());
		}
	}
}

void CGeometriesEditorTab::AddGeometryStd(EVolumeShape _shape, EType _type)
{
	if (_type == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;
	const SVolumeType domain = m_pSystemStructure->GetBoundingBox();
	const CVector3 center = (domain.coordBeg + domain.coordEnd) / 2;
	const CGeometrySizes sizes = CGeometrySizes::Defaults(Length(domain.coordEnd - domain.coordBeg));
	const CBaseGeometry* geometry{ nullptr };
	switch (_type)
	{
	case EType::NONE: return;
	case EType::GEOMETRY:	geometry = m_pSystemStructure->AddGeometry(_shape, sizes, center);			break;
	case EType::VOLUME:		geometry = m_pSystemStructure->AddAnalysisVolume(_shape, sizes, center);	break;
	}

	if (!geometry) return;
	UpdateGeometriesList();
	ui.listGeometries->SetCurrentItem(QString::fromStdString(geometry->Key()));
	EmitChangeSignals();
}

void CGeometriesEditorTab::AddGeometryLib(const std::string& _key, EType _type)
{
	if (_type == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;
	const CBaseGeometry* geometry{ nullptr };
	switch (_type)
	{
	case EType::NONE: return;
	case EType::GEOMETRY:	geometry = m_pSystemStructure->AddGeometry(m_pGeometriesDB->Geometry(_key)->mesh);			break;
	case EType::VOLUME:		geometry = m_pSystemStructure->AddAnalysisVolume(m_pGeometriesDB->Geometry(_key)->mesh);	break;
	}

	if (!geometry) return;
	UpdateGeometriesList();
	ui.listGeometries->SetCurrentItem(QString::fromStdString(geometry->Key()));
	EmitChangeSignals();
}

void CGeometriesEditorTab::DeleteGeometry()
{
	const EType type = Type();
	if (type == EType::NONE) return;

	if (QMessageBox::question(this, windowTitle(), tr("Delete %1 %2?").arg(type == EType::GEOMETRY ? "geometry" : "volume").arg(QString::fromStdString(m_object->Name())), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	// find an item to select after deleting of the current one
	auto toSelect = CQtTree::GetData(ui.listGeometries->itemAbove(ui.listGeometries->currentItem()));
	if(toSelect.isEmpty())
		toSelect = CQtTree::GetData(ui.listGeometries->itemBelow(ui.listGeometries->currentItem()));
	// delete
	if (type == EType::GEOMETRY)	m_pSystemStructure->DeleteGeometry(m_object->Key());
	else							m_pSystemStructure->DeleteAnalysisVolume(m_object->Key());

	EmitChangeSignals(type);
	UpdateGeometriesList();
	ui.listGeometries->SetCurrentItem(toSelect);
}

void CGeometriesEditorTab::UpGeometry()
{
	switch (Type())
	{
	case EType::NONE: return;
	case EType::GEOMETRY:	m_pSystemStructure->UpGeometry(m_object->Key());		break;
	case EType::VOLUME:		m_pSystemStructure->UpAnalysisVolume(m_object->Key());	break;
	}

	EmitChangeSignals();
	UpdateGeometriesList();
}

void CGeometriesEditorTab::DownGeometry()
{
	switch (Type())
	{
	case EType::NONE: return;
	case EType::GEOMETRY:	m_pSystemStructure->DownGeometry(m_object->Key());			break;
	case EType::VOLUME:		m_pSystemStructure->DownAnalysisVolume(m_object->Key());	break;
	}

	EmitChangeSignals();
	UpdateGeometriesList();
}

void CGeometriesEditorTab::NameChanged()
{
	if (!m_object) return;

	const QString name = ui.listGeometries->currentItem()->text(0);
	if (name.isEmpty()) return;

	m_object->SetName(name.toStdString());

	EmitChangeSignals();
	UpdateGeometriesList();
	UpdatePropertiesInfo();
}

void CGeometriesEditorTab::GeometrySelected()
{
	if (auto* item = ui.listGeometries->currentItem())
	{
		if (item->parent() == m_list[EType::GEOMETRY])		m_object = m_pSystemStructure->Geometry(CQtTree::GetData(item).toStdString());
		else if (item->parent() == m_list[EType::VOLUME])	m_object = m_pSystemStructure->AnalysisVolume(CQtTree::GetData(item).toStdString());
		else												m_object = nullptr;
	}
	else
		m_object = nullptr;

	UpdateMotionCombo();
	UpdatePropertiesInfo();
	UpdateMotionInfo();
}

void CGeometriesEditorTab::MaterialChanged()
{
	if (Type() != EType::GEOMETRY) return;
	const auto key = ui.treeProperties->GetComboBoxValue(m_properties.at(EProperty::MATERIAL), 1).toString().toStdString();
	dynamic_cast<CRealGeometry*>(m_object)->SetMaterial(key);
	emit ObjectsChanged();
}

void CGeometriesEditorTab::ColorChanged()
{
	const CColor color = ui.treeProperties->GetColorPickerValue(m_properties.at(EProperty::COLOR), 1);
	m_object->SetColor(color);
	EmitChangeSignals();
}

void CGeometriesEditorTab::MotionTypeChanged()
{
	const CGeometryMotion::EMotionType type = static_cast<CGeometryMotion::EMotionType>(ui.treeProperties->GetComboBoxValue(m_properties.at(EProperty::MOTION), 1).toUInt());
	m_object->Motion()->SetMotionType(type);
	UpdateMotionInfo();
}

void CGeometriesEditorTab::RotationChanged(EAxis _axis)
{
	if (!m_object) return;
	if (Type() == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;

	CVector3 angle{ 0.0 };
	angle[E2I(_axis)] = ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(static_cast<EProperty>(EProperty::ROTATE_X + E2I(_axis))), 1);
	angle *= PI / 180;
	const CMatrix3 rotationMatrix = CQuaternion{ angle }.ToRotmat();
	m_object->Rotate(rotationMatrix);

	EmitChangeSignals();
	UpdatePropertiesInfo();
}

void CGeometriesEditorTab::PositionChanged()
{
	if (!m_object) return;
	if (Type() == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;

	const CVector3 center{
		ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_X), 1),
		ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_Y), 1),
		ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::POSITION_Z), 1)	};
	m_object->SetCenter(center);

	EmitChangeSignals();
	UpdatePropertiesInfo();
}

void CGeometriesEditorTab::SizeChanged()
{
	if (!m_object) return;
	if (Type() == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;

	CGeometrySizes sizes;
	if (m_object->Shape() == EVolumeShape::VOLUME_STL)
	{
		sizes.SetWidth( ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_X), 1));
		sizes.SetDepth( ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_Y), 1));
		sizes.SetHeight(ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::SIZE_Z), 1));
	}
	else
	{
		sizes.SetWidth(      ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::WIDTH),        1));
		sizes.SetDepth(      ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::DEPTH),        1));
		sizes.SetHeight(     ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::HEIGHT),       1));
		sizes.SetRadius(     ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::RADIUS),       1));
		sizes.SetInnerRadius(ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::INNER_RADIUS), 1));
	}
	m_object->Resize(sizes);

	EmitChangeSignals();
	UpdatePropertiesInfo();
}

void CGeometriesEditorTab::QualityChanged()
{
	if (!m_object) return;
	if (Type() == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;
	m_object->SetAccuracy(CMeshGenerator::TrianglesToAccuracy(m_object->Shape(), ui.treeProperties->GetListSpinBoxValue(m_properties.at(EProperty::TRIANGLES), 1)));
	EmitChangeSignals();
	UpdatePropertiesInfo();
}

void CGeometriesEditorTab::ScalingChanged()
{
	if (!m_object) return;
	if (Type() == EType::GEOMETRY && !CheckAndConfirmTPRemoval()) return;

	const double factor = ui.treeProperties->GetDoubleSpinBoxValue(m_properties.at(EProperty::SCALE), 1);
	if (factor <= 0) return;
	m_object->Scale(factor);

	EmitChangeSignals();
	UpdatePropertiesInfo();
}

void CGeometriesEditorTab::AddMotion()
{
	if (!m_object) return;
	m_object->Motion()->AddInterval();
	UpdateMotionInfo();
}

void CGeometriesEditorTab::DeleteMotion()
{
	if (!m_object) return;

	std::set<int> cols;
	for (const auto& index : ui.tableMotion->selectionModel()->selection().indexes())
		cols.insert(index.column());
	for (auto i = cols.rbegin(); i != cols.rend(); ++i)
		m_object->Motion()->DeleteInterval(*i);

	UpdateMotionInfo();
}

void CGeometriesEditorTab::MotionTableChanged()
{
	if (!m_object) return;

	for (int iCol = 0; iCol < ui.tableMotion->columnCount(); ++iCol)
		switch (m_object->Motion()->MotionType())
		{
		case CGeometryMotion::EMotionType::TIME_DEPENDENT:
			m_object->Motion()->ChangeTimeInterval(iCol, {
				ui.tableMotion->GetConvValue(ERowMotion::TIME_BEG, iCol, EUnitType::TIME),
				ui.tableMotion->GetConvValue(ERowMotion::TIME_END, iCol, EUnitType::TIME), {
				ui.tableMotion->GetConvVectorCol(ERowMotion::VEL_X, iCol, EUnitType::VELOCITY),
				Type() == EType::GEOMETRY ? ui.tableMotion->GetConvVectorCol(ERowMotion::ROT_VEL_X, iCol, EUnitType::ANGULAR_VELOCITY) : CVector3{ 0.0 },
				Type() == EType::GEOMETRY ? ui.tableMotion->GetConvVectorCol(ERowMotion::ROT_CENTER_X, iCol, EUnitType::LENGTH) : CVector3{ 0.0 }
				} });
			break;
		case CGeometryMotion::EMotionType::FORCE_DEPENDENT:
		case CGeometryMotion::EMotionType::CONSTANT_FORCE:
			m_object->Motion()->ChangeForceInterval(iCol, {
				ui.tableMotion->GetConvValue(ERowMotion::FORCE, iCol, EUnitType::FORCE),
				static_cast<CGeometryMotion::SForceMotionInterval::ELimitType>(ui.tableMotion->GetComboBoxValue(ERowMotion::LIMIT_TYPE, iCol).toUInt()), {
				ui.tableMotion->GetConvVectorCol(ERowMotion::VEL_X, iCol, EUnitType::VELOCITY),
				ui.tableMotion->GetConvVectorCol(ERowMotion::ROT_VEL_X, iCol, EUnitType::ANGULAR_VELOCITY) ,
				ui.tableMotion->GetConvVectorCol(ERowMotion::ROT_CENTER_X, iCol, EUnitType::LENGTH)
				} });
			break;
		case CGeometryMotion::EMotionType::NONE: break;
		}

	UpdateMotionInfo();
}

void CGeometriesEditorTab::MotionChanged()
{
	if (Type() != EType::GEOMETRY) return;

	auto* geometry = dynamic_cast<CRealGeometry*>(m_object);
	geometry->SetRotateAroundCenter(ui.checkBoxAroundCenter->isChecked());
	geometry->SetMass(GetConvValue(ui.lineEditMass, EUnitType::MASS));
	geometry->SetFreeMotion(CBasicVector3<bool>{ ui.checkBoxFreeMotionX->isChecked(), ui.checkBoxFreeMotionY->isChecked(), ui.checkBoxFreeMotionZ->isChecked() });

	UpdateMotionInfo();
}

void CGeometriesEditorTab::PlaceAside(EAxis _axis, EPosition _pos)
{
	if (!m_object || m_pSystemStructure->GetAllSpheres(0, true).empty()) return;

	// index of the entry (x/y/z) in CVector3
	const size_t e = E2I(_axis);

	// choose proper functions
	using fun_type = double(&)(std::initializer_list<double>);
	fun_type& fun1 = _pos == EPosition::MIN ? static_cast<fun_type>(std::min) : static_cast<fun_type>(std::max);
	fun_type& fun2 = _pos == EPosition::MAX ? static_cast<fun_type>(std::min) : static_cast<fun_type>(std::max);

	// min/max position of existing particles
	double partPos = fun2({ -std::numeric_limits<double>::max(), std::numeric_limits<double>::max() });
	for (const auto& part : m_pSystemStructure->GetAllSpheres(0, true))
		partPos = fun1({ partPos, part->GetCoordinates(0)[e] + part->GetRadius() * (_pos == EPosition::MIN ? -1 : 1) });

	// min/max position of walls of the given geometry
	const auto bb = m_object->BoundingBox();
	const double wallPos = fun2({ bb.coordBeg[e], bb.coordEnd[e] });

	// calculate offset for geometry
	const CVector3 offset{
		_axis == EAxis::X ? partPos - wallPos : 0,
		_axis == EAxis::Y ? partPos - wallPos : 0,
		_axis == EAxis::Z ? partPos - wallPos : 0 };

	// shift geometry
	m_object->Shift(offset);

	EmitChangeSignals();
	UpdatePropertiesInfo();
}

CGeometriesEditorTab::EType CGeometriesEditorTab::Type() const
{
	if (dynamic_cast<CRealGeometry*>(m_object)) return EType::GEOMETRY;
	if (dynamic_cast<CAnalysisVolume*>(m_object)) return EType::VOLUME;
	return EType::NONE;
}

void CGeometriesEditorTab::EmitChangeSignals(EType _type/* = EType::NONE*/)
{
	emit ObjectsChanged();
	if (_type == EType::VOLUME || _type == EType::NONE && dynamic_cast<CAnalysisVolume*>(m_object))
		emit AnalysisGeometriesChanged();
}

bool CGeometriesEditorTab::CheckAndConfirmTPRemoval()
{
	if (m_pSystemStructure->GetMaxTime() <= 0) return true;
	const auto answer = QMessageBox::question(this, windowTitle(), "This scene contains time-dependent data. All time points after 0 will be removed. Continue?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
	if (answer != QMessageBox::Yes) return false;
	m_pSystemStructure->ClearAllStatesFrom(0);
	return true;
}

void CGeometriesEditorTab::keyPressEvent(QKeyEvent* _event)
{
	switch (_event->key())
	{
	case Qt::Key_Delete:
		if (ui.listGeometries->hasFocus()) DeleteGeometry(); break;
	default: CMusenDialog::keyPressEvent(_event);
	}
}
