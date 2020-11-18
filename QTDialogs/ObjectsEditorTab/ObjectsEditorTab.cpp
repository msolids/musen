/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ObjectsEditorTab.h"
#include "ViewSettings.h"
#include "qtOperations.h"
#include <QMenu>
#include "QtSignalBlocker.h"

CObjectsEditorTab::CObjectsEditorTab(CExportTDPTab* _pTab, CViewSettings* _viewSettings, QSettings* _settings, QWidget* parent) :
	CMusenDialog(parent),
	m_settings{ _settings },
	m_exportTDPTab{ _pTab },
	m_viewSettings{ _viewSettings }
{
	ui.setupUi(this);
	setWindowFlags(Qt::Window);

	ui.objectsTable->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
	ui.objectsTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	setContextMenuPolicy(Qt::CustomContextMenu);

	m_dataFieldsPart = {
		{ EFieldTypes::ID,					"ID",						{ "ID" },													},
		{ EFieldTypes::MATERIAL,			"Material",					{ "Material" },												},
		{ EFieldTypes::COORDINATE,			"Coordinate",				{ "X", "Y", "Z" },			EUnitType::LENGTH,				},
		{ EFieldTypes::VELOCITY,			"Velocity",					{ "Vx", "Vy", "Vz" },		EUnitType::VELOCITY,			},
		{ EFieldTypes::ROTATION_VELOCITY,	"Rotational velocity",		{ "Wx", "Wy", "Wz" },		EUnitType::ANGULAR_VELOCITY,	},
		{ EFieldTypes::DIAMETER,			"Diameter",					{ "D" },					EUnitType::PARTICLE_DIAMETER,	},
		{ EFieldTypes::CONTACT_DIAMETER,	"Contact D",				{ "Contact D" },			EUnitType::PARTICLE_DIAMETER,	},
		{ EFieldTypes::COORDINATION_NUMBER,	"Coordination number",		{ "Coord number" },											},
		{ EFieldTypes::MAX_OVERLAP,	        "Max overlap",		        { "Max overlap" },			EUnitType::LENGTH,				},
		{ EFieldTypes::FORCE,				"Force",					{ "Fx", "Fy", "Fz" },		EUnitType::FORCE,				},
		{ EFieldTypes::STRESS,				"Stress",					{ "Sx", "Sy", "Sz" },		EUnitType::PRESSURE,			},
		{ EFieldTypes::TEMPERATURE,			"Temperature",				{ "T" },					EUnitType::TEMPERATURE,			},
		{ EFieldTypes::ORIENTATION,			"Orientation",				{ "Qx", "Qy", "Qz", "Qw" },									},
	};

	m_dataFieldsBond = {
		{ EFieldTypes::ID,					"ID",						{ "ID" },												    },
		{ EFieldTypes::MATERIAL,			"Material",					{ "Material" },											    },
		{ EFieldTypes::COORDINATE,			"Coordinate",				{ "X", "Y", "Z" },			EUnitType::LENGTH,			    },
		{ EFieldTypes::VELOCITY,			"Velocity",					{ "Vx", "Vy", "Vz" },		EUnitType::VELOCITY,		    },
		{ EFieldTypes::PARTNERS_ID,			"Particles ID",				{ "ID1", "ID2" },										    },
		{ EFieldTypes::DIAMETER,			"Diameter",					{ "D" },					EUnitType::PARTICLE_DIAMETER    },
		{ EFieldTypes::LENGTH,				"Length",					{ "L" },					EUnitType::LENGTH,			    },
		{ EFieldTypes::INITIAL_LENGTH,		"Initial length",			{ "Initial L" },			EUnitType::LENGTH,			    },
		{ EFieldTypes::FORCE,				"Force",					{ "Fx", "Fy", "Fz" },		EUnitType::FORCE,			    },
		{ EFieldTypes::TEMPERATURE,			"Temperature",				{ "T" },					EUnitType::TEMPERATURE,		    },
		{ EFieldTypes::TANGENTIAL_OVERLAP,	"Tang overlap",				{ "Overl X", "Overl Y", "Overl Z"  },		EUnitType::LENGTH,		    },
	};

	m_dataFieldsWall = {
		{ EFieldTypes::ID,					"ID",						{ "ID" },								                    },
		{ EFieldTypes::MATERIAL,			"Material",					{ "Material" },							                    },
		{ EFieldTypes::GEOMETRY,			"Geometry",					{ "Geometry" },							                    },
		{ EFieldTypes::COORDINATE,			"Coordinate",				{ "X1", "Y1", "Z1",
																		  "X2", "Y2", "Z2",
																		  "X3", "Y3", "Z3" },		EUnitType::LENGTH	            },
		{ EFieldTypes::NORMAL,				"Normal vector",			{ "Nx", "Ny", "Nz" },		EUnitType::LENGTH	            },
		{ EFieldTypes::VELOCITY,			"Velocity",					{ "Vx", "Vy", "Vz" },		EUnitType::VELOCITY	            },
		{ EFieldTypes::FORCE,				"Force",					{ "Fx", "Fy", "Fz" },		EUnitType::FORCE	            },
	};

	ui.objectsTable->BlockingPaste(true);
	ui.groupBoxAddObject->setVisible(false);

	LoadConfiguration();
	SetupObjectTypesCombo();
	UpdateVisibleFields();
	InitializeConnections();
}

void CObjectsEditorTab::Initialize()
{
	ui.objectsTable->SetUnitConverter(m_pUnitConverter);
	ui.newObjectPanel->SetPointers(m_pSystemStructure, m_pMaterialsDB, m_pUnitConverter);
}

void CObjectsEditorTab::InitializeConnections() const
{
	connect(ui.objectTypes, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &CObjectsEditorTab::ObjectTypeChanged);

	connect(ui.objectsTable, &CQtTable::itemSelectionChanged,	this, &CObjectsEditorTab::ObjectsSelectionChanged);
	connect(ui.objectsTable, &CQtTable::itemChanged,			this, &CObjectsEditorTab::ObjectDataChanged);
	connect(ui.objectsTable, &CQtTable::DataPasted,				this, &CObjectsEditorTab::DataPasted);

	connect(ui.buttonUpdate,	   &QPushButton::clicked, this, &CObjectsEditorTab::UpdateButtonPressed);
	connect(ui.checkBoxAutoUpdate, &QPushButton::toggled, this, &CObjectsEditorTab::AutoUpdateToggled);

	connect(this, &CObjectsEditorTab::customContextMenuRequested, this, &CObjectsEditorTab::ShowContextMenu);

	connect(ui.showHidePanelButton, &QPushButton::clicked,			this, &CObjectsEditorTab::ToggleAddPanelVisibility);
	connect(ui.newObjectPanel,		&CNewObjectPanel::ObjectAdded,	this, &CObjectsEditorTab::ObjectAdded);
}

void CObjectsEditorTab::SetupObjectTypesCombo() const
{
	CQtSignalBlocker blocker{ ui.objectTypes };

	ui.objectTypes->insertItem(EObjectTypes::PARTICLES,		"Particles",	EObjectTypes::PARTICLES);
	ui.objectTypes->insertItem(EObjectTypes::SOLID_BONDS,	"Solid bonds",	EObjectTypes::SOLID_BONDS);
	ui.objectTypes->insertItem(EObjectTypes::WALLS,			"Walls",		EObjectTypes::WALLS);
	ui.objectTypes->setCurrentIndex(m_currentObjectsType);
}

void CObjectsEditorTab::UpdateWholeView()
{
	if (!isVisible()) return;
	UpdateTimeView();
	UpdateVisibleFields();
	TryUpdateTable();
	UpdateAutoButtonBlock();
	ui.newObjectPanel->UpdateWholeView();
}

void CObjectsEditorTab::UpdateTimeView() const
{
	ShowConvLabel(ui.timeLabel, "Time", EUnitType::TIME);
	ShowConvValue(ui.currentTime, m_dCurrentTime, EUnitType::TIME);
}

void CObjectsEditorTab::UpdateVisibleFields()
{
	CQtSignalBlocker blocker{ ui.groupBoxDataFields };

	// clear old checkboxes
	while (auto* w = ui.groupBoxDataFields->findChild<QWidget*>())
		delete w;

	// add checkboxes
	for (const auto& field : CurrentDataFields())
	{
		auto* box = new QCheckBox{ QString::fromStdString(field.name), ui.groupBoxDataFields };
		box->setChecked(field.active);
		box->setProperty("ID", static_cast<int>(field.type));
		ui.groupBoxDataFields->layout()->addWidget(box);
		connect(box, &QCheckBox::toggled, this, [=] { FieldActivityChanged(box); });
	}
}

void CObjectsEditorTab::UpdateAutoButtonBlock() const
{
	CQtSignalBlocker blocker{ ui.buttonUpdate, ui.checkBoxAutoUpdate };
	ui.buttonUpdate->setEnabled(!m_autoUpdate);
	ui.checkBoxAutoUpdate->setChecked(m_autoUpdate);
}

void CObjectsEditorTab::TryUpdateTable()
{
	if (m_autoUpdate)
		UpdateTable();
	else
		ui.objectsTable->setEnabled(false);
}

void CObjectsEditorTab::UpdateTable()
{
	CQtSignalBlocker blocker{ ui.objectsTable };
	ui.objectsTable->setSortingEnabled(false);

	// count number of columns
	size_t columns = 0;
	for (const auto& field : ActiveDataFields())
		columns += field.headers.size();
	ui.objectsTable->setRowCount(0); // is needed for speed-up
	ui.objectsTable->setColumnCount(static_cast<int>(columns));
	ui.objectsTable->setEnabled(true);

	// set headers
	int iCol = 0;
	for (const auto& field : ActiveDataFields())
		for (const auto& header : field.headers)
			ui.objectsTable->SetColHeaderItemConv(iCol++, header, field.units);

	switch (SelectedObjectsType())
	{
	case PARTICLES:		UpdateTableParts();	break;
	case SOLID_BONDS:	UpdateTableBonds();	break;
	case WALLS:			UpdateTableWalls();	break;
	}

	ui.objectsTable->setSortingEnabled(true);

	UpdateSelectedObjects();

	raise();
}

void CObjectsEditorTab::UpdateTableParts() const
{
	const auto particles = m_pSystemStructure->GetAllSpheres(m_dCurrentTime, false);
	const int number = static_cast<int>(particles.size());
	m_pSystemStructure->PrepareTimePointForRead(m_dCurrentTime);
	ui.objectsTable->setRowCount(number);

	int iCol = 0;
	for (const auto& field : ActiveDataFields())
	{
		switch (field.type)
		{
		case EFieldTypes::ID:
		{
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, particles[iRow]->m_lObjectID);
			break;
		}
		case EFieldTypes::MATERIAL:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, m_pMaterialsDB->GetCompoundName(particles[iRow]->GetCompoundKey()));
			break;
		case EFieldTypes::COORDINATE:
		{
			std::vector<CVector3> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetCoordinates();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::VELOCITY:
		{
			std::vector<CVector3> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetVelocity();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::ROTATION_VELOCITY:
		{
			std::vector<CVector3> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetAngleVelocity();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::DIAMETER:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemEditableConv(iRow, iCol, particles[iRow]->GetRadius() * 2, field.units);
			break;
		case EFieldTypes::CONTACT_DIAMETER:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemEditableConv(iRow, iCol, particles[iRow]->GetContactRadius() * 2, field.units);
			break;
		case EFieldTypes::COORDINATION_NUMBER:
		{
			const std::vector<unsigned> coordinations = m_pSystemStructure->GetCoordinationNumbers(m_dCurrentTime);
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, coordinations[particles[iRow]->m_lObjectID]);
			break;
		}
		case EFieldTypes::MAX_OVERLAP:
		{
			const std::vector<double> overlaps = m_pSystemStructure->GetMaxOverlaps(m_dCurrentTime);
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditableConv(iRow, iCol, overlaps[particles[iRow]->m_lObjectID], field.units);
			break;
		}
		case EFieldTypes::FORCE:
		{
			std::vector<CVector3> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetForce();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::STRESS:
		{
			std::vector<CVector3> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetNormalStress();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::TEMPERATURE:
		{
			std::vector<double> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetTemperature();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::ORIENTATION:
		{
			std::vector<CQuaternion> data(particles.size());
			ParallelFor(particles.size(), [&](size_t iRow)
			{
				data[iRow] = particles[iRow]->GetOrientation();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::PARTNERS_ID:
		case EFieldTypes::LENGTH:
		case EFieldTypes::INITIAL_LENGTH:
		case EFieldTypes::NORMAL:
		case EFieldTypes::TANGENTIAL_OVERLAP:
		case EFieldTypes::GEOMETRY: break;
		}

		iCol += static_cast<int>(field.headers.size());
	}

	// show not active objects gray
	for (int iRow = 0; iRow < number; ++iRow)
		if (!particles[iRow]->IsActive(m_dCurrentTime))
			ui.objectsTable->SetRowBackgroundColor(InactiveTableColor(), iRow);

	// set IDs of objects to the first row to be able to identify them
	if (ui.objectsTable->columnCount())
		for (int iRow = 0; iRow < number; ++iRow)
			ui.objectsTable->SetItemUserData(iRow, 0, particles[iRow]->m_lObjectID);
}

void CObjectsEditorTab::UpdateTableBonds() const
{
	const auto bonds = m_pSystemStructure->GetAllSolidBonds(m_dCurrentTime, false);
	const int number = static_cast<int>(bonds.size());
	m_pSystemStructure->PrepareTimePointForRead(m_dCurrentTime);
	ui.objectsTable->setRowCount(number);

	int iCol = 0;
	for (const auto& field : ActiveDataFields())
	{
		switch (field.type)
		{
		case EFieldTypes::ID:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, bonds[iRow]->m_lObjectID);
			break;
		case EFieldTypes::MATERIAL:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, m_pMaterialsDB->GetCompoundName(bonds[iRow]->GetCompoundKey()));
			break;
		case EFieldTypes::COORDINATE:
		{
			std::vector<CVector3> data(bonds.size());
			ParallelFor(bonds.size(), [&](size_t iRow)
			{
				data[iRow] = m_pSystemStructure->GetBondCoordinate(m_dCurrentTime, bonds[iRow]->m_lObjectID);
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::VELOCITY:
		{
			std::vector<CVector3> data(bonds.size());
			ParallelFor(bonds.size(), [&](size_t iRow)
			{
				data[iRow] = m_pSystemStructure->GetBondVelocity(m_dCurrentTime, bonds[iRow]->m_lObjectID);
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::PARTNERS_ID:
			for (int iRow = 0; iRow < number; ++iRow)
			{
				ui.objectsTable->SetItemEditable(iRow, iCol + 0, bonds[iRow]->m_nLeftObjectID);
				ui.objectsTable->SetItemEditable(iRow, iCol + 1, bonds[iRow]->m_nRightObjectID);
			}
			break;
		case EFieldTypes::DIAMETER:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemEditableConv(iRow, iCol, bonds[iRow]->GetDiameter(), field.units);
			break;
		case EFieldTypes::LENGTH:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditableConv(iRow, iCol, m_pSystemStructure->GetBond(m_dCurrentTime, bonds[iRow]->m_lObjectID).Length(), field.units);
			break;
		case EFieldTypes::INITIAL_LENGTH:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemEditableConv(iRow, iCol, bonds[iRow]->GetInitLength(), field.units);
			break;
		case EFieldTypes::FORCE:
		{
			std::vector<CVector3> data(bonds.size());
			ParallelFor(bonds.size(), [&](size_t iRow)
			{
				data[iRow] = bonds[iRow]->GetForce();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::TEMPERATURE:
		{
			std::vector<double> data(bonds.size());
			ParallelFor(bonds.size(), [&](size_t iRow)
			{
				data[iRow] = bonds[iRow]->GetTemperature();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::TANGENTIAL_OVERLAP:
		{
			std::vector<CVector3> data(bonds.size());
			ParallelFor(bonds.size(), [&](size_t iRow)
			{
				data[iRow] = bonds[iRow]->GetTangentialOverlap();
			});
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowEditableConv(iRow, iCol, data[iRow], field.units);
			break;
		}
		case EFieldTypes::MAX_OVERLAP:
		case EFieldTypes::ROTATION_VELOCITY:
		case EFieldTypes::CONTACT_DIAMETER:
		case EFieldTypes::ORIENTATION:
		case EFieldTypes::COORDINATION_NUMBER:
		case EFieldTypes::NORMAL:
		case EFieldTypes::GEOMETRY:
		case EFieldTypes::STRESS: break;
		}

		iCol += static_cast<int>(field.headers.size());
	}

	// show not active objects gray
	for (int iRow = 0; iRow < number; ++iRow)
		if (!bonds[iRow]->IsActive(m_dCurrentTime))
			ui.objectsTable->SetRowBackgroundColor(InactiveTableColor(), iRow);

	// set IDs of objects to the first row to be able to identify them
	if (ui.objectsTable->columnCount())
		for (int iRow = 0; iRow < number; ++iRow)
			ui.objectsTable->SetItemUserData(iRow, 0, bonds[iRow]->m_lObjectID);
}

void CObjectsEditorTab::UpdateTableWalls() const
{
	const auto walls = m_pSystemStructure->GetAllWalls(m_dCurrentTime, false);
	const int number = static_cast<int>(walls.size());
	m_pSystemStructure->PrepareTimePointForRead(m_dCurrentTime);
	ui.objectsTable->setRowCount(number);

	int iCol = 0;
	for (const auto& field : ActiveDataFields())
	{
		switch (field.type)
		{
		case EFieldTypes::ID:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, walls[iRow]->m_lObjectID);
			break;
		case EFieldTypes::MATERIAL:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, m_pMaterialsDB->GetCompoundName(walls[iRow]->GetCompoundKey()));
			break;
		case EFieldTypes::GEOMETRY:
		{
			std::vector<size_t> geometryID(m_pSystemStructure->GetTotalObjectsCount());
			for (size_t iGeometry = 0; iGeometry < m_pSystemStructure->GeometriesNumber(); ++iGeometry)
				for (const auto& plane : m_pSystemStructure->Geometry(iGeometry)->Planes())
					geometryID[plane] = iGeometry;
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemNotEditable(iRow, iCol, m_pSystemStructure->Geometry(geometryID[walls[iRow]->m_lObjectID])->Name());
			break;
		}
		case EFieldTypes::COORDINATE:
			for (int iRow = 0; iRow < number; ++iRow)
			{
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol + 0, walls[iRow]->GetCoordVertex1(), field.units);
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol + 3, walls[iRow]->GetCoordVertex2(), field.units);
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol + 6, walls[iRow]->GetCoordVertex3(), field.units);
			}
			break;
		case EFieldTypes::NORMAL:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, walls[iRow]->GetNormalVector(), field.units);
			break;
		case EFieldTypes::VELOCITY:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, walls[iRow]->GetVelocity(), field.units);
			break;
		case EFieldTypes::FORCE:
			for (int iRow = 0; iRow < number; ++iRow)
				ui.objectsTable->SetItemsRowNotEditableConv(iRow, iCol, walls[iRow]->GetForce(), field.units);
			break;
		case EFieldTypes::TEMPERATURE:
		case EFieldTypes::ORIENTATION:
		case EFieldTypes::ROTATION_VELOCITY:
		case EFieldTypes::DIAMETER:
		case EFieldTypes::CONTACT_DIAMETER:
		case EFieldTypes::COORDINATION_NUMBER:
		case EFieldTypes::MAX_OVERLAP:
		case EFieldTypes::PARTNERS_ID:
		case EFieldTypes::LENGTH:
		case EFieldTypes::INITIAL_LENGTH:
		case EFieldTypes::TANGENTIAL_OVERLAP:
		case EFieldTypes::STRESS: break;
		}

		iCol += static_cast<int>(field.headers.size());
	}

	// show not active objects gray
	for (int iRow = 0; iRow < number; ++iRow)
		if (!walls[iRow]->IsActive(m_dCurrentTime))
			ui.objectsTable->SetRowBackgroundColor(InactiveTableColor(), iRow);

	// set IDs of objects to the first row to be able to identify them
	if (ui.objectsTable->columnCount())
		for (int iRow = 0; iRow < number; ++iRow)
			ui.objectsTable->SetItemUserData(iRow, 0, walls[iRow]->m_lObjectID);
}

void CObjectsEditorTab::SetObjectData(int _row) const
{
	if (ui.objectsTable->columnCount() == 0) return;
	CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(ui.objectsTable->GetItemUserData(_row, 0).toUInt());
	if (!object) return;

	CQtSignalBlocker blocker{ ui.objectsTable };
	m_pSystemStructure->PrepareTimePointForWrite(m_dCurrentTime);

	switch (SelectedObjectsType())
	{
	case PARTICLES:		SetObjectDataPart(_row, *dynamic_cast<CSphere*>(object));			break;
	case SOLID_BONDS:	SetObjectDataBond(_row, *dynamic_cast<CSolidBond*>(object));		break;
	case WALLS:			SetObjectDataWall(_row, *dynamic_cast<CTriangularWall*>(object));	break;
	}
}

void CObjectsEditorTab::SetObjectDataPart(int _row, CSphere& _part) const
{
	int iCol = 0;
	for (const auto& field : ActiveDataFields())
	{
		switch (field.type)
		{
		case EFieldTypes::COORDINATE:			_part.SetCoordinates(ui.objectsTable->GetConvVectorRow(_row, iCol, field.units));	break;
		case EFieldTypes::VELOCITY:				_part.SetVelocity(ui.objectsTable->GetConvVectorRow(_row, iCol, field.units));		break;
		case EFieldTypes::ROTATION_VELOCITY:	_part.SetAngleVelocity(ui.objectsTable->GetConvVectorRow(_row, iCol, field.units));	break;
		case EFieldTypes::DIAMETER:				_part.SetRadius(ui.objectsTable->GetConvValue(_row, iCol, field.units) / 2);		break;
		case EFieldTypes::CONTACT_DIAMETER:		_part.SetContactRadius(ui.objectsTable->GetConvValue(_row, iCol, field.units) / 2);	break;
		case EFieldTypes::TEMPERATURE:			_part.SetTemperature(ui.objectsTable->GetConvValue(_row, iCol, field.units));		break;
		case EFieldTypes::ORIENTATION:			_part.SetOrientation(ui.objectsTable->GetConvQuartRow(_row, iCol, field.units));	break;
		case EFieldTypes::ID:
		case EFieldTypes::MATERIAL:
		case EFieldTypes::COORDINATION_NUMBER:
		case EFieldTypes::MAX_OVERLAP:
		case EFieldTypes::FORCE:
		case EFieldTypes::PARTNERS_ID:
		case EFieldTypes::LENGTH:
		case EFieldTypes::INITIAL_LENGTH:
		case EFieldTypes::NORMAL:
		case EFieldTypes::GEOMETRY:
		case EFieldTypes::TANGENTIAL_OVERLAP:
		case EFieldTypes::STRESS: break;
		}

		iCol += static_cast<int>(field.headers.size());
	}
}

void CObjectsEditorTab::SetObjectDataBond(int _row, CSolidBond& _bond) const
{
	int iCol = 0;
	for (const auto& field : ActiveDataFields())
	{
		switch (field.type)
		{
		case EFieldTypes::PARTNERS_ID:
		{
			// get new particle IDs
			const unsigned newL = ui.objectsTable->item(_row, iCol + 0)->text().toUInt();
			const unsigned newR = ui.objectsTable->item(_row, iCol + 1)->text().toUInt();
			// get contact partners
			auto *part1 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(newL));
			auto *part2 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(newR));
			// check that contact partners exist
			if (!part1 || !part2)
			{
				ui.statusMessage->setText("Error: Invalid IDs of contact partners!");
				// set old values
				ui.objectsTable->SetItemEditable(_row, iCol + 0, _bond.m_nLeftObjectID);
				ui.objectsTable->SetItemEditable(_row, iCol + 1, _bond.m_nRightObjectID);
				break;
			}
			// set new contact partners
			_bond.m_nLeftObjectID = newL;
			_bond.m_nRightObjectID = newR;
			// update initial length
			_bond.SetInitialLength(m_pSystemStructure->GetBond(m_dCurrentTime, _bond.m_lObjectID).Length());
			break;
		}
		case EFieldTypes::DIAMETER:
		{
			// get new diameter
			const double newD = ui.objectsTable->GetConvValue(_row, iCol, EUnitType::PARTICLE_DIAMETER);
			// get contact partners
			auto *part1 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(_bond.m_nLeftObjectID));
			auto *part2 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(_bond.m_nRightObjectID));
			// check that the new diameter is not larger then the contact partners
			if (part1->GetRadius() < newD / 2 || part2->GetRadius() < newD / 2)
			{
				ui.statusMessage->setText("Error: Bond diameter is larger than the diameter of the contact partners!");
				// set old value
				ui.objectsTable->SetItemEditableConv(_row, iCol, _bond.GetDiameter(), field.units);
				break;
			}
			// set new diameter
			_bond.SetDiameter(newD);
			break;
		}
		case EFieldTypes::TANGENTIAL_OVERLAP: _bond.SetTangentialOverlap(ui.objectsTable->GetConvVectorRow(_row, iCol, field.units)); break;
		case EFieldTypes::INITIAL_LENGTH:	_bond.SetInitialLength(ui.objectsTable->GetConvValue(_row, iCol, field.units));	break;
		case EFieldTypes::TEMPERATURE:		_bond.SetTemperature(ui.objectsTable->GetConvValue(_row, iCol, field.units));	break;
		case EFieldTypes::ID:
		case EFieldTypes::MATERIAL:
		case EFieldTypes::COORDINATE:
		case EFieldTypes::VELOCITY:
		case EFieldTypes::LENGTH:
		case EFieldTypes::FORCE:
		case EFieldTypes::ROTATION_VELOCITY:
		case EFieldTypes::CONTACT_DIAMETER:
		case EFieldTypes::MAX_OVERLAP:
		case EFieldTypes::ORIENTATION:
		case EFieldTypes::COORDINATION_NUMBER:
		case EFieldTypes::NORMAL:
		case EFieldTypes::GEOMETRY:
		case EFieldTypes::STRESS: break;
		}

		iCol += static_cast<int>(field.headers.size());
	}
}

void CObjectsEditorTab::SetObjectDataWall(int _row, CTriangularWall& _wall) const
{
	int iCol = 0;
	for (const auto& field : ActiveDataFields())
	{
		switch (field.type)
		{
		case EFieldTypes::COORDINATE:
		case EFieldTypes::VELOCITY:
		case EFieldTypes::ROTATION_VELOCITY:
		case EFieldTypes::DIAMETER:
		case EFieldTypes::CONTACT_DIAMETER:
		case EFieldTypes::TEMPERATURE:
		case EFieldTypes::ORIENTATION:
		case EFieldTypes::ID:
		case EFieldTypes::MATERIAL:
		case EFieldTypes::COORDINATION_NUMBER:
		case EFieldTypes::MAX_OVERLAP:
		case EFieldTypes::FORCE:
		case EFieldTypes::PARTNERS_ID:
		case EFieldTypes::LENGTH:
		case EFieldTypes::INITIAL_LENGTH:
		case EFieldTypes::NORMAL:
		case EFieldTypes::GEOMETRY:
		case EFieldTypes::TANGENTIAL_OVERLAP:
		case EFieldTypes::STRESS: break;
		}

		iCol += static_cast<int>(field.headers.size());
	}
}

void CObjectsEditorTab::ObjectTypeChanged()
{
	m_currentObjectsType = static_cast<EObjectTypes>(ui.objectTypes->currentData().toInt());
	UpdateVisibleFields();
	TryUpdateTable();
}

void CObjectsEditorTab::FieldActivityChanged(const QCheckBox* _checkbox)
{
	// change activity of the data field
	const EFieldTypes type = static_cast<EFieldTypes>(_checkbox->property("ID").toInt());
	for (auto& field : CurrentDataFields())
		if (field.type == type)
		{
			field.active = _checkbox->isChecked();
			break;
		}

	TryUpdateTable();
}

void CObjectsEditorTab::UpdateButtonPressed()
{
	UpdateTable();
}

void CObjectsEditorTab::AutoUpdateToggled()
{
	m_autoUpdate = ui.checkBoxAutoUpdate->isChecked();
	UpdateAutoButtonBlock();
	if (m_autoUpdate)
		UpdateTable();
}

CObjectsEditorTab::EObjectTypes CObjectsEditorTab::SelectedObjectsType() const
{
	return m_currentObjectsType;
}

std::vector<CObjectsEditorTab::SDataField>& CObjectsEditorTab::CurrentDataFields()
{
	switch (SelectedObjectsType())
	{
	case PARTICLES:		return m_dataFieldsPart;
	case SOLID_BONDS:	return m_dataFieldsBond;
	case WALLS:			return m_dataFieldsWall;
	}

	return m_dataFieldsPart; // should never reach this
}

const std::vector<CObjectsEditorTab::SDataField>& CObjectsEditorTab::CurrentDataFields() const
{
	return const_cast<CObjectsEditorTab*>(this)->CurrentDataFields();
}

std::vector<CObjectsEditorTab::SDataField> CObjectsEditorTab::ActiveDataFields() const
{
	std::vector<SDataField> res;
	res.reserve(m_dataFieldsPart.size()); // just because it is the longest
	for (const auto& field : CurrentDataFields())
		if (field.active)
			res.push_back(field);
	return res;
}

void CObjectsEditorTab::UpdateSelectedObjects() const
{
	if (!isVisible()) return;
	CQtSignalBlocker blocker{ ui.objectsTable };

	// get objects to select
	const auto& selectedIDs = m_viewSettings->SelectedObjects();

	// clear current selection
	QItemSelectionModel* selectionModel = ui.objectsTable->selectionModel();
	selectionModel->clearSelection();

	// if there is nothing to select
	if (selectedIDs.empty() || ui.objectsTable->columnCount() == 0) return;

	// get selection object
	QItemSelection itemSelection = selectionModel->selection();

	// collect pairs <objectID, row> for all items
	std::map<size_t, int> idInRow;
	for (int i = 0; i < ui.objectsTable->rowCount(); ++i)
		idInRow[ui.objectsTable->GetItemUserData(i, 0).toUInt()] = i;

	// gather selection of objects
	std::vector<std::pair<QPersistentModelIndex, QPersistentModelIndex>> indices;
	indices.reserve(selectedIDs.size());
	for (const auto id : selectedIDs)
		if (MapContainsKey(idInRow, id))
		{
			const auto row = idInRow[id];
			const QPersistentModelIndex topL = ui.objectsTable->model()->index(row, 0);
			const QPersistentModelIndex botR = ui.objectsTable->model()->index(row, ui.objectsTable->columnCount() - 1);
			if (!indices.empty() && indices.back().second.row() == row - 1)
				indices.back().second = botR;
			else
				indices.emplace_back(topL, botR);
		}
	std::vector<QItemSelection> selections(GetThreadsNumber(), itemSelection);
	ParallelFor([&](size_t iThread)
	{
		for (size_t i = iThread; i < indices.size(); i += GetThreadsNumber())
			selections[iThread].merge(QItemSelection{ indices[i].first, indices[i].second }, QItemSelectionModel::Select);
	});
	for (const auto& selection : selections)
		itemSelection.merge(selection, QItemSelectionModel::Select);

	// apply selection
	selectionModel->select(itemSelection, QItemSelectionModel::Select);

	// scroll to the first selected item and set focus to the table to highlight selection
	if (!itemSelection.indexes().isEmpty())
		ui.objectsTable->scrollTo(itemSelection.indexes().front());
	ui.objectsTable->setFocus(Qt::ActiveWindowFocusReason);
}

void CObjectsEditorTab::ObjectsSelectionChanged() const
{
	std::vector<size_t> objects;
	const QList<QTableWidgetSelectionRange> selection = ui.objectsTable->selectedRanges();
	for (int i = 0; i < selection.count(); ++i)
		for (int j = 0; j < selection.at(i).rowCount(); ++j)
			objects.push_back(ui.objectsTable->GetItemUserData(selection.at(i).topRow() + j, 0).toUInt());
	m_viewSettings->SelectedObjects(objects);
	emit ObjectsSelected();
}

void CObjectsEditorTab::DeleteSelectedObjects()
{
	CQtSignalBlocker blocker{ ui.objectsTable };

	m_pSystemStructure->DeleteObjects(m_viewSettings->SelectedObjects());
	m_viewSettings->SelectedObjects({});

	UpdateTable();
	emit ObjectsSelected();
	emit MaterialsChanged();
	emit UpdateOpenGLView();
}

void CObjectsEditorTab::ObjectDataChanged(QTableWidgetItem* _item)
{
	ui.statusMessage->clear();
	SetObjectData(_item->row());
	FitBondsDiameters();
	FitPartsContactRadii();
	emit UpdateOpenGLView();
}

void CObjectsEditorTab::DataPasted() const
{
	const CQtTable::SPasteInfo modified = ui.objectsTable->GetModifiedRows();
	const int rowsNumber = modified.lastModifiedRow - modified.firstModifiedRow;
	for (int i = 0; i < rowsNumber; ++i)
		SetObjectData(i + modified.firstModifiedRow);
	FitBondsDiameters();
	FitPartsContactRadii();
	ui.statusMessage->setText("Data for " + QString::number(rowsNumber) + " objects was changed.");
}

void CObjectsEditorTab::FitBondsDiameters() const
{
	size_t counter = 0;
	for (auto& bond : m_pSystemStructure->GetAllSolidBonds(0, false))
	{
		auto* part1 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(bond->m_nLeftObjectID));
		auto* part2 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(bond->m_nRightObjectID));
		if (!part1 || !part2) continue;
		if (part1->GetRadius() < bond->GetDiameter() / 2 || part2->GetRadius() < bond->GetDiameter() / 2)
		{
			bond->SetDiameter(std::min({ bond->GetDiameter(), 2 * part1->GetRadius(), 2 * part2->GetRadius() }));
			counter++;
		}
	}

	if (counter)
		ui.statusMessage->setText("Diameter of " + QString::number(counter) + " bonds has been respectively reduced.");
}

void CObjectsEditorTab::FitPartsContactRadii() const
{
	if (m_pSystemStructure->IsContactRadiusEnabled()) return;

	for (auto& part : m_pSystemStructure->GetAllSpheres(0, false))
		if (part->GetRadius() != part->GetContactRadius())
			part->SetContactRadius(part->GetRadius());
}

void CObjectsEditorTab::ShowContextMenu(const QPoint& _pos)
{
	const auto GetContextValue = [&](const QString& _titel, const QString& _label, EUnitType _units) -> std::pair<bool, double>
	{
		bool okPressed;
		const double value = QInputDialog::getText(this, _titel, _label + " [" + QString::fromStdString(m_pUnitConverter->GetSelectedUnit(_units)) + "]",
			QLineEdit::Normal, "1e-3", &okPressed).toDouble();
		return { okPressed, m_pUnitConverter->GetValueSI(_units, value) };
	};

	if (QApplication::focusWidget() != ui.objectsTable) return;

	QMenu contextMenu;
	QAction* actionExportTDP = contextMenu.addAction("Export TDP");
	QAction* actionSetDiameter = contextMenu.addAction("Set new diameter");
	actionSetDiameter->setEnabled(SelectedObjectsType() == PARTICLES || SelectedObjectsType() == SOLID_BONDS);

	auto* velocityMenu = new QMenu("Set new velocity", this);
	velocityMenu->setEnabled(SelectedObjectsType() == PARTICLES);
	QAction* actionSetVelX = velocityMenu->addAction("Set new velocity X");
	QAction* actionSetVelY = velocityMenu->addAction("Set new velocity Y");
	QAction* actionSetVelZ = velocityMenu->addAction("Set new velocity Z");
	contextMenu.addMenu(velocityMenu);

	auto* materialsMenu = new QMenu("Set new material", this);
	for (size_t i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
	{
		auto* action = new QAction(QString::fromStdString(m_pMaterialsDB->GetCompoundName(i)), this);
		materialsMenu->addAction(action);
		connect(action, &QAction::triggered, this, [=] { SetNewMaterial(i); });
	}
	contextMenu.addMenu(materialsMenu);

	const QAction* selectedAction = contextMenu.exec(mapToGlobal(_pos));
	if (selectedAction == actionExportTDP)
		ShowTDPExportTab();
	else if (selectedAction == actionSetDiameter)
	{
		const auto res = GetContextValue("Specify new diameter", "New diameter", EUnitType::PARTICLE_DIAMETER);
		if (res.first)	SetNewDiameter(res.second);
	}
	else if (selectedAction == actionSetVelX)
	{
		const auto res = GetContextValue("Specify new X velocity", "New velocity", EUnitType::VELOCITY);
		if (res.first)	SetNewVelocity(res.second, EDirection::X);
	}
	else if (selectedAction == actionSetVelY)
	{
		const auto res = GetContextValue("Specify new Y velocity", "New velocity", EUnitType::VELOCITY);
		if (res.first)	SetNewVelocity(res.second, EDirection::Y);
	}
	else if (selectedAction == actionSetVelZ)
	{
		const auto res = GetContextValue("Specify new Z velocity", "New velocity", EUnitType::VELOCITY);
		if (res.first)	SetNewVelocity(res.second, EDirection::Z);
	}
}

void CObjectsEditorTab::ShowTDPExportTab() const
{
	m_exportTDPTab->SetSelectedObjectsID(m_viewSettings->SelectedObjects());
	m_exportTDPTab->show();
}

void CObjectsEditorTab::SetNewDiameter(double _diameter)
{
	for (auto id : m_viewSettings->SelectedObjects())
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(id);
		if (auto* sphere = dynamic_cast<CSphere*>(object))
		{
			sphere->SetRadius(_diameter / 2);
			sphere->SetContactRadius(_diameter / 2);
		}
		else if (auto* bond = dynamic_cast<CBond*>(object))
			bond->SetDiameter(_diameter);
	}
	FitBondsDiameters();
	TryUpdateTable();
	emit UpdateOpenGLView();
}

void CObjectsEditorTab::SetNewVelocity(double _velocity, EDirection _direction)
{
	for (auto id : m_viewSettings->SelectedObjects())
	{
		CVector3 vel = m_pSystemStructure->GetObjectByIndex(id)->GetVelocity(m_dCurrentTime);
		switch (_direction)
		{
		case EDirection::X: vel.x = _velocity; break;
		case EDirection::Y: vel.y = _velocity; break;
		case EDirection::Z: vel.z = _velocity; break;
		}
		m_pSystemStructure->GetObjectByIndex(id)->SetVelocity(m_dCurrentTime, vel);
	}
	TryUpdateTable();
	emit UpdateOpenGLView();
}

void CObjectsEditorTab::SetNewMaterial(size_t _iCompound)
{
	const CCompound* compound = m_pMaterialsDB->GetCompound(_iCompound);
	if (!compound) return;
	const std::string key = compound->GetKey();
	for (auto id : m_viewSettings->SelectedObjects())
		m_pSystemStructure->GetObjectByIndex(id)->SetCompoundKey(key);
	m_pSystemStructure->UpdateObjectsCompoundProperties(m_viewSettings->SelectedObjects());
	TryUpdateTable();
	emit MaterialsChanged();
	emit UpdateOpenGLView();
}

void CObjectsEditorTab::ToggleAddPanelVisibility() const
{
	if (ui.groupBoxAddObject->isVisible())
		ui.showHidePanelButton->setText("Show Object panel" );
	else
		ui.showHidePanelButton->setText("Hide Object panel");
	ui.groupBoxAddObject->setVisible(!ui.groupBoxAddObject->isVisible());
}

void CObjectsEditorTab::ObjectAdded()
{
	ui.statusMessage->setText(ui.newObjectPanel->StatusMessage());
	UpdateWholeView();
	emit MaterialsChanged();
	emit UpdateOpenGLView();
}

void CObjectsEditorTab::SetEditEnabled(bool _enable) const
{
	ui.groupBoxAddObject->setEnabled(_enable);
	if (_enable)
		ui.objectsTable->setEditTriggers(QAbstractItemView::EditTrigger::DoubleClicked | QAbstractItemView::EditTrigger::EditKeyPressed | QAbstractItemView::EditTrigger::AnyKeyPressed);
	else
		ui.objectsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
}

void CObjectsEditorTab::SaveConfiguration() const
{
	const auto SaveActiveTypes = [&](const QString& _key, const std::vector<SDataField>& _dataFields)
	{
		QList<int> activeTypes;
		for (const auto& field : _dataFields)
			if (field.active)
				activeTypes.append(static_cast<int>(field.type));
		m_settings->setValue(_key, QVariant::fromValue(activeTypes));
	};

	m_settings->setValue(c_AUTO_UPDATE       , m_autoUpdate);
	m_settings->setValue(c_OBJECTS_TYPE      , m_currentObjectsType);
	SaveActiveTypes(c_FIELDS_PARTICLES  , m_dataFieldsPart);
	SaveActiveTypes(c_FIELDS_SOLID_BONDS, m_dataFieldsBond);
	SaveActiveTypes(c_FIELDS_WALLS      , m_dataFieldsWall);
}

void CObjectsEditorTab::LoadConfiguration()
{
	const auto LoadActiveTypes = [&](const QString& _key, std::vector<SDataField>& _dataFields)
	{
		if (m_settings->value(_key).isValid())
		{
			const QList<int>& activeTypes = m_settings->value(_key).value<QList<int>>();
			for (auto& field : _dataFields)
				field.active = activeTypes.contains(static_cast<int>(field.type));
		}
	};

	if (m_settings->value(c_AUTO_UPDATE).isValid())		m_autoUpdate = m_settings->value(c_AUTO_UPDATE).toBool();
	if (m_settings->value(c_OBJECTS_TYPE).isValid())	m_currentObjectsType = static_cast<EObjectTypes>(m_settings->value(c_OBJECTS_TYPE).toInt());
	LoadActiveTypes(c_FIELDS_PARTICLES  , m_dataFieldsPart);
	LoadActiveTypes(c_FIELDS_SOLID_BONDS, m_dataFieldsBond);
	LoadActiveTypes(c_FIELDS_WALLS      , m_dataFieldsWall);
}

void CObjectsEditorTab::keyPressEvent(QKeyEvent* _event)
{
	switch (_event->key())
	{
	case Qt::Key_F5:
		UpdateWholeView();
		break;
	case Qt::Key_Delete:
		if (QApplication::focusWidget() == ui.objectsTable)
			DeleteSelectedObjects();
		break;
	default: CMusenDialog::keyPressEvent(_event);
	}
}
