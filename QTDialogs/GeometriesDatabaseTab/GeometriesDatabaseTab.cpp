/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesDatabaseTab.h"
#include "QtSignalBlocker.h"
#include <QFileDialog>
#include <QLockFile>
#include <QMenu>
#include <QMessageBox>
#include <QShortcut>

CGeometriesDatabaseTab::CGeometriesDatabaseTab(QWidget *parent)
	: CMusenDialog(parent)
{
	ui.setupUi(this);

	ui.tableGeometryInfo->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeMode::Stretch);
	ui.tableGeometryInfo->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeMode::Stretch);

	ui.viewer->SetAxes(false);
	ui.viewer->SetLegend({}, false);
	ui.viewer->SetTime({}, false);

	SetupScaleButton();
	InitializeConnections();

	m_sHelpFileName = "Users Guide/Geometries Database.pdf";
}

void CGeometriesDatabaseTab::InitializeConnections()
{
	connect(ui.buttonNewDatabase,	 &QPushButton::clicked, this, &CGeometriesDatabaseTab::NewDatabase);
	connect(ui.buttonLoadDatabase,	 &QPushButton::clicked, this, &CGeometriesDatabaseTab::LoadDatabase);
	connect(ui.buttonSaveDatabase,	 &QPushButton::clicked, this, &CGeometriesDatabaseTab::SaveDatabase);
	connect(ui.ButtonSaveAsDatabase, &QPushButton::clicked, this, &CGeometriesDatabaseTab::SaveDatabaseAs);

	connect(ui.buttonImportGeometry, &QPushButton::clicked, this, &CGeometriesDatabaseTab::ImportGeometry);
	connect(ui.buttonDeleteGeometry, &QPushButton::clicked, this, &CGeometriesDatabaseTab::DeleteGeometry);
	connect(ui.buttonUpGeometry,     &QPushButton::clicked, this, &CGeometriesDatabaseTab::UpGeometry);
	connect(ui.buttonDownGeometry,   &QPushButton::clicked, this, &CGeometriesDatabaseTab::DownGeometry);
	connect(ui.buttonExportGeometry, &QPushButton::clicked, this, &CGeometriesDatabaseTab::ExportGeometry);

	connect(ui.listGeometries, &QListWidget::currentItemChanged, this, &CGeometriesDatabaseTab::GeometrySelected);
	connect(ui.listGeometries, &QListWidget::itemChanged,        this, &CGeometriesDatabaseTab::GeometryRenamed);

	connect(new QShortcut{ QKeySequence{ "Ctrl+N" },       this }, &QShortcut::activated, this, &CGeometriesDatabaseTab::NewDatabase);
	connect(new QShortcut{ QKeySequence{ "Ctrl+O" },       this }, &QShortcut::activated, this, &CGeometriesDatabaseTab::LoadDatabase);
	connect(new QShortcut{ QKeySequence{ "Ctrl+S" },       this }, &QShortcut::activated, this, &CGeometriesDatabaseTab::SaveDatabase);
	connect(new QShortcut{ QKeySequence{ "Ctrl+Shift+S" }, this }, &QShortcut::activated, this, &CGeometriesDatabaseTab::SaveDatabaseAs);
}

void CGeometriesDatabaseTab::Initialize()
{
	ui.tableGeometryInfo->SetUnitConverter(m_pUnitConverter);
}

void CGeometriesDatabaseTab::SetupScaleButton()
{
	auto* scaleMenu = new QMenu(this);
	auto* action_m  = new QAction("to [m]",  this);
	auto* action_mm = new QAction("to [mm]", this);
	auto* action_um = new QAction("to [um]", this);
	auto* action_nm = new QAction("to [nm]", this);
	scaleMenu->addAction(action_m);
	scaleMenu->addAction(action_mm);
	scaleMenu->addAction(action_um);
	scaleMenu->addAction(action_nm);
	connect(action_m,  &QAction::triggered, this, [=]() { ScaleGeometry(1.0); });
	connect(action_mm, &QAction::triggered, this, [=]() { ScaleGeometry(1e-3); });
	connect(action_um, &QAction::triggered, this, [=]() { ScaleGeometry(1e-6); });
	connect(action_nm, &QAction::triggered, this, [=]() { ScaleGeometry(1e-9); });
	ui.buttonScaleGeometry->setMenu(scaleMenu);
}

void CGeometriesDatabaseTab::NewDatabase()
{
	m_pGeometriesDB->NewDatabase();
	SetDBModified(true);
	UpdateWholeView();
}

void CGeometriesDatabaseTab::LoadDatabase()
{
	const QString fileName = QFileDialog::getOpenFileName(this, tr("Load geometries database"), DefaultPath(), tr("MUSEN geometries database (*.mgdb);;All files (*.*);;"));
	if (fileName.simplified().isEmpty()) return;
	m_pGeometriesDB->LoadFromFile(fileName.toStdString());
	m_lastUsedFilePath = fileName;
	SetDBModified(false);
	UpdateWholeView();
}

void CGeometriesDatabaseTab::SaveDatabase()
{
	const std::string currFile = m_pGeometriesDB->GetFileName();
	if (!currFile.empty())
	{
		m_pGeometriesDB->SaveToFile(currFile);
		SetDBModified(false);
		UpdateWindowTitle();
	}
	else
		SaveDatabaseAs();
}

void CGeometriesDatabaseTab::SaveDatabaseAs()
{
	const QString defaultFileName = !m_pGeometriesDB->GetFileName().empty() ? QString::fromStdString(m_pGeometriesDB->GetFileName()) : "Geometries";
	const QString fileName = QFileDialog::getSaveFileName(this, tr("Save geometries database"), DefaultPath() + defaultFileName, tr("MUSEN geometries database (*.mgdb);;All files (*.*);;"));
	if (fileName.simplified().isEmpty()) return;
	m_pGeometriesDB->SaveToFile(fileName.toStdString());
	m_lastUsedFilePath = fileName;
	SetDBModified(false);
	UpdateWindowTitle();
}

void CGeometriesDatabaseTab::ImportGeometry()
{
	const QString fileName = QFileDialog::getOpenFileName(this, tr("Import from STL file"), DefaultPath(), tr("STL files (*.stl);;All files (*.*);;"));
	if (fileName.simplified().isEmpty()) return;
	m_pGeometriesDB->AddGeometry(fileName.toStdString());
	m_lastUsedFilePath = fileName;
	SetDBModified(true);
	UpdateGeometriesList();
	ui.listGeometries->setCurrentRow(ui.listGeometries->count() - 1); // select added geometry
	emit GeometryAdded();
}

void CGeometriesDatabaseTab::ExportGeometry()
{
	const auto geometry = m_pGeometriesDB->Geometry(ui.listGeometries->currentRow());
	if (!geometry) return;
	const QString defaultFileName = QString::fromStdString(geometry->mesh.Name());
	const QString fileName = QFileDialog::getSaveFileName(this, tr("Export to STL file"), DefaultPath() + defaultFileName, tr("STL files (*.stl);;All files (*.*);;"));
	if (fileName.simplified().isEmpty()) return;
	m_pGeometriesDB->ExportGeometry(ui.listGeometries->currentRow(), fileName.toStdString());
	m_lastUsedFilePath = fileName;
}

void CGeometriesDatabaseTab::DeleteGeometry()
{
	m_pGeometriesDB->DeleteGeometry(ui.listGeometries->currentRow());
	SetDBModified(true);
	UpdateGeometriesList();
}

void CGeometriesDatabaseTab::UpGeometry()
{
	const int oldRow = ui.listGeometries->currentRow();
	m_pGeometriesDB->UpGeometry(oldRow);
	SetDBModified(true);
	UpdateGeometriesList();
	ui.listGeometries->setCurrentRow(oldRow == 0 ? 0 : oldRow - 1);
}

void CGeometriesDatabaseTab::DownGeometry()
{
	const int oldRow = ui.listGeometries->currentRow();
	const int lastRow = ui.listGeometries->count() - 1;
	m_pGeometriesDB->DownGeometry(oldRow);
	SetDBModified(true);
	UpdateGeometriesList();
	ui.listGeometries->setCurrentRow(oldRow == lastRow ? lastRow : oldRow + 1);
}

void CGeometriesDatabaseTab::ScaleGeometry(double _factor)
{
	m_pGeometriesDB->ScaleGeometry(ui.listGeometries->currentRow(), _factor);
	SetDBModified(true);
	UpdateGeometryInfo();
}

void CGeometriesDatabaseTab::UpdateWholeView()
{
	UpdateWindowTitle();
	UpdateButtons();
	UpdateGeometryInfoHeaders();
	UpdateGeometriesList();
}

void CGeometriesDatabaseTab::UpdateWindowTitle()
{
	const QString titlePrefix = "Geometries database: ";
	const QString titleSuffix = !m_pGeometriesDB->GetFileName().empty() ? QString::fromStdString(m_pGeometriesDB->GetFileName()) + "[*]" : "-";
	setWindowTitle(titlePrefix + titleSuffix);
}

void CGeometriesDatabaseTab::UpdateGeometriesList() const
{
	CQtSignalBlocker blocker{ ui.listGeometries };
	const int oldRow = ui.listGeometries->currentRow();

	ui.listGeometries->clear();
	for (const auto& geometry : m_pGeometriesDB->Geometries())
	{
		auto item = new QListWidgetItem(QString::fromStdString(geometry->mesh.Name()));
		item->setFlags(item->flags() | Qt::ItemIsEditable);
		ui.listGeometries->addItem(item);
	}

	ui.listGeometries->RestoreCurrentRow(oldRow);

	Update3DView();
}

void CGeometriesDatabaseTab::UpdateGeometryInfoHeaders() const
{
	ui.tableGeometryInfo->SetColHeaderItemConv(1, "Length (X)", EUnitType::LENGTH);
	ui.tableGeometryInfo->SetColHeaderItemConv(2, "Depth (Y)" , EUnitType::LENGTH);
	ui.tableGeometryInfo->SetColHeaderItemConv(3, "Height (Z)", EUnitType::LENGTH);
}

void CGeometriesDatabaseTab::UpdateGeometryInfo() const
{
	const auto* geometry = m_pGeometriesDB->Geometry(ui.listGeometries->currentRow());
	ui.tableGeometryInfo->setEnabled(geometry != nullptr);
	if (geometry)
	{
		const SVolumeType bb = geometry->mesh.BoundingBox();
		ui.tableGeometryInfo->SetItemNotEditable(0, 0, QString::number(geometry->mesh.TrianglesNumber()));
		ui.tableGeometryInfo->SetItemNotEditableConv(1, 0, bb.coordEnd.x - bb.coordBeg.x, EUnitType::LENGTH);
		ui.tableGeometryInfo->SetItemNotEditableConv(2, 0, bb.coordEnd.y - bb.coordBeg.y, EUnitType::LENGTH);
		ui.tableGeometryInfo->SetItemNotEditableConv(3, 0, bb.coordEnd.z - bb.coordBeg.z, EUnitType::LENGTH);
	}
	else
		ui.tableGeometryInfo->clearContents();
}

void CGeometriesDatabaseTab::Update3DView() const
{
	// a functions to convert data to the viewer's format
	const auto C2Q = [](const CVector3& _v)
	{
		return QVector3D{ static_cast<float>(_v.x), static_cast<float>(_v.y), static_cast<float>(_v.z) };
	};

	const auto* geometry = m_pGeometriesDB->Geometry(ui.listGeometries->currentRow());

	// create a copy of the mesh and put it in the origin
	CTriangularMesh mesh{ geometry ? geometry->mesh : CTriangularMesh{} };
	mesh.SetCenter(CVector3{ 0, 0, 0 });
	// transform triangles to the viewer's format
	std::vector<COpenGLViewShader::STriangle> triangles;
	triangles.reserve(mesh.TrianglesNumber());
	for (const auto& t : mesh.Triangles())
		triangles.emplace_back(COpenGLViewShader::STriangle{ C2Q(t.p1), C2Q(t.p2), C2Q(t.p3), QColor{127, 127, 255, 255} });
	const auto bb = mesh.BoundingBox();
	// set data to the viewer
	ui.viewer->SetWalls(triangles);
	// TODO: remove it here
	ui.viewer->SetAxes(false);
	ui.viewer->SetCameraStandardView(SBox{ C2Q(bb.coordBeg), C2Q(bb.coordEnd) }, C2Q(CVector3{ 0, -1, 0 }));
	ui.viewer->Redraw();
}

void CGeometriesDatabaseTab::UpdateButtons() const
{
	if (!m_pGeometriesDB) return;
	// check whether it is possible to write into current file
	const QString lockFileName = QString::fromStdString(m_pGeometriesDB->GetFileName()) + ".lock";
	QLockFile fileLocker{ lockFileName };
	fileLocker.setStaleLockTime(0);
	const bool locked = fileLocker.tryLock(10);
	fileLocker.unlock();
	ui.buttonSaveDatabase->setEnabled(locked);
}

void CGeometriesDatabaseTab::GeometrySelected() const
{
	const auto* geometry = m_pGeometriesDB->Geometry(ui.listGeometries->currentRow());
	ui.buttonDeleteGeometry->setEnabled(geometry != nullptr);
	UpdateGeometryInfo();
	Update3DView();
}

void CGeometriesDatabaseTab::GeometryRenamed()
{
	auto* geometry = m_pGeometriesDB->Geometry(ui.listGeometries->currentRow());
	if (!geometry) return;
	geometry->mesh.SetName(ui.listGeometries->currentItem()->text().toStdString());
	SetDBModified(true);
}

QString CGeometriesDatabaseTab::DefaultPath() const
{
	if (!m_lastUsedFilePath.isEmpty()) return QString::fromStdString(MUSENFileFunctions::FilePath(m_lastUsedFilePath.toStdString())) + "/";
	const std::string fileName = !m_pGeometriesDB->GetFileName().empty() ? m_pGeometriesDB->GetFileName() : m_pSystemStructure->GetFileName();
	return QString::fromStdString(MUSENFileFunctions::FilePath(fileName)) + "/";
}

void CGeometriesDatabaseTab::SetDBModified(bool _modified)
{
	m_isDBModified = _modified;
	setWindowModified(_modified);
}

void CGeometriesDatabaseTab::keyPressEvent(QKeyEvent* _event)
{
	switch (_event->key())
	{
	case Qt::Key_Delete:
		if (ui.listGeometries->hasFocus())
			DeleteGeometry();
		break;
	default: CMusenDialog::keyPressEvent(_event);
	}
}

void CGeometriesDatabaseTab::closeEvent(QCloseEvent* _event)
{
	// TODO: add it after switch to Qt 5.10+ on Linux.
	// check if any item is currently editing
	//if (ui.listGeometries->isPersistentEditorOpen(ui.listGeometries->currentItem()))
	//	setFocus(); // force to finish editing

	if (m_isDBModified)
	{
		const QMessageBox::StandardButtons buttons = QMessageBox::Yes | QMessageBox::Cancel | QMessageBox::No;
		const QMessageBox::StandardButton reply = QMessageBox::question(this, "Geometries database", "Save changes to geometries database", buttons);
		if (reply == QMessageBox::Yes)
			SaveDatabase();
		else if (reply == QMessageBox::Cancel)
		{
			_event->ignore();
			return;
		}
	}
	CMusenDialog::closeEvent(_event);
}