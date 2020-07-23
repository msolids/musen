/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesDatabaseTab.h"

CGeometriesDatabaseTab::CGeometriesDatabaseTab( QWidget *parent )
	: CMusenDialog( parent )
{
	ui.setupUi(this);
	ui.selGeometryTable->horizontalHeader()->setSectionResizeMode( QHeaderView::ResizeMode::Stretch );
	ui.selGeometryTable->verticalHeader()->setSectionResizeMode( QHeaderView::ResizeMode::Stretch );

	m_sHelpFileName = "Users Guide/Geometries Database.pdf";
	InitializeConnections();
	for ( int i = 0; i<ui.selGeometryTable->rowCount();i++ )
		ui.selGeometryTable->item( i, 0 )->setFlags( ui.selGeometryTable->item( i, 0 )->flags() & (~Qt::ItemIsEditable) );
}

void CGeometriesDatabaseTab::InitializeConnections()
{
	connect(ui.addGeometry, SIGNAL(clicked()), this, SLOT(AddNewGeometry()));
	connect(ui.pushButtonExport, SIGNAL(clicked()), this, SLOT(ExportGeometry()));
	connect( ui.newDatabase, SIGNAL( clicked() ), this, SLOT( NewDatabase() ) );
	connect( ui.saveDatabase, SIGNAL(clicked()), this, SLOT(SaveDatabase()) );
	connect( ui.saveDatabaseAs, SIGNAL( clicked() ), this, SLOT( SaveDatabaseAs() ) );
	connect( ui.loadDatabase, SIGNAL( clicked() ), this, SLOT( LoadDatabase() ) );
	connect( ui.deleteGeometry, SIGNAL( clicked() ), this, SLOT( DeleteGeometry() ) );

	connect( ui.geometriesList, SIGNAL( currentItemChanged( QListWidgetItem*, QListWidgetItem* ) ), this, SLOT( NewRowSelected() ) );
	connect( ui.geometriesList, SIGNAL( itemChanged( QListWidgetItem* ) ), this, SLOT( DataWasChanged() ) );

	connect( ui.upGeometry, SIGNAL( clicked()), this, SLOT( UpGeometry() ) );
	connect( ui.downGeometry, SIGNAL( clicked()), this, SLOT (DownGeometry()));
}

void CGeometriesDatabaseTab::UpGeometry()
{
	int oldRow = ui.geometriesList->currentRow();
	m_pGeometriesDB->UpGeometry(oldRow);
	UpdateWholeView();
	ui.geometriesList->setCurrentRow( oldRow == 0 ? 0 : --oldRow );
}

void CGeometriesDatabaseTab::DownGeometry()
{
	int oldRow = ui.geometriesList->currentRow();
	int lastRow = ui.geometriesList->count() - 1;
	m_pGeometriesDB->DownGeometry(oldRow);
	UpdateWholeView();
	ui.geometriesList->setCurrentRow( oldRow == lastRow ? lastRow : ++oldRow );
}

void CGeometriesDatabaseTab::DeleteGeometry()
{
	m_pGeometriesDB->DeleteGeometry( ui.geometriesList->currentRow() );
	UpdateWholeView();
}

void CGeometriesDatabaseTab::AddNewGeometry()
{
	QString sFileName = QFileDialog::getOpenFileName( this, tr( "Select STL file" ), "", tr( "STL files (*.stl);;All files (*.*);;" ) );
	if ( sFileName.isEmpty() ) return;

	m_pGeometriesDB->AddGeometry( qs2ss( sFileName ) );
	UpdateWholeView();
}

void CGeometriesDatabaseTab::ExportGeometry()
{
	QString sFileName = QFileDialog::getSaveFileName(this, tr("Save geometry"), "", tr("STL files (*.stl);;All files (*.*);;"));
	if (sFileName.isEmpty()) return;

	int index = ui.geometriesList->currentRow();
	if (index <= 0) return;

	m_pGeometriesDB->ExportGeometry((size_t)index, qs2ss(sFileName));
}

void CGeometriesDatabaseTab::ShowGeometry()
{
	int nIndex = ui.geometriesList->currentRow();
	const CTriangularMesh* pGeom = m_pGeometriesDB->GetGeometry(nIndex);
	if ( !pGeom ) return;
	ui.geomOpenGLView->SetCurrentGeometry(pGeom->sName, pGeom->vTriangles);
}

void CGeometriesDatabaseTab::NewDatabase()
{
	m_pGeometriesDB->NewDatabase();
	UpdateWholeView();
}

void CGeometriesDatabaseTab::DataWasChanged()
{
	if ( m_bAvoidSignal ) return;
	for ( int i = 0; i< ui.geometriesList->count(); i++ )
	{
		CTriangularMesh* pGeom = m_pGeometriesDB->GetGeometry( i );
		if ( !pGeom ) continue;
		pGeom->sName = qs2ss( ui.geometriesList->item( i )->text() );
	}
	UpdateWholeView();
}

void CGeometriesDatabaseTab::LoadDatabase()
{
	QString sFileName = QFileDialog::getOpenFileName(this, tr("Load geometries database"), "", tr( "Process (*.mgdb);;All files (*.*);;" ));
	if ( sFileName.simplified() == "" )
		return;
	m_pGeometriesDB->LoadFromFile( qs2ss( sFileName ) );
	UpdateWholeView();
}

void CGeometriesDatabaseTab::SaveDatabase()
{
	SaveDatabaseAs( ss2qs( m_pGeometriesDB->GetFileName() ) );
}

void CGeometriesDatabaseTab::SaveDatabaseAs( const QString& _sFileName /*= "" */ )
{
	QString sFileName = _sFileName;
	if ( sFileName.simplified() == "" )
		sFileName = QFileDialog::getSaveFileName(this, tr("Save geometries database"), "", tr( "Processes (*.mgdb);;All files (*.*);;" ));
	if ( sFileName.simplified() == "" )
		return;
	m_pGeometriesDB->SaveToFile( qs2ss( sFileName ) );
	UpdateWholeView();
}

void CGeometriesDatabaseTab::UpdateWholeView()
{
	m_bAvoidSignal = true;
	ui.geometriesList->clear();
	UpdateButtons();
	setWindowTitle( "Geometries database: " + ss2qs( m_pGeometriesDB->GetFileName() ) );

	for ( unsigned i = 0; i< m_pGeometriesDB->GetGeometriesNumber(); i++ )
	{
		CTriangularMesh* pGeom = m_pGeometriesDB->GetGeometry( i );
		if ( !pGeom ) continue;
		ui.geometriesList->insertItem( i, new QListWidgetItem( ss2qs( pGeom->sName ) ) );
		ui.geometriesList->item( i )->setFlags( ui.geometriesList->item( i )->flags() | Qt::ItemIsEditable );
	}
	m_bAvoidSignal = false;
	if ( (ui.geometriesList->currentRow() == -1) && (m_pGeometriesDB->GetGeometriesNumber() > 0) )
		ui.geometriesList->setCurrentRow( 0 );
	NewRowSelected();
	UpdateSelectedGeomInfo();
}

void CGeometriesDatabaseTab::NewRowSelected()
{
	int nCurrentRow = ui.geometriesList->currentRow();
	CTriangularMesh* pGeom = NULL;
	if (nCurrentRow >= 0)
		pGeom = m_pGeometriesDB->GetGeometry(nCurrentRow);
	ui.deleteGeometry->setEnabled( pGeom != NULL );
	if(isVisible())
		ShowGeometry();
	UpdateSelectedGeomInfo();
}

void CGeometriesDatabaseTab::UpdateSelectedGeomInfo()
{
	int nCurrentRow = ui.geometriesList->currentRow();
	CTriangularMesh* pGeom = NULL;
	if ( nCurrentRow >= 0 )
		pGeom = m_pGeometriesDB->GetGeometry( nCurrentRow );
	ui.selGeometryTable->setEnabled( pGeom != NULL );
	if ( pGeom != NULL )
	{
		ui.selGeometryTable->item( 0, 0 )->setText( ss2qs( pGeom->sName ) );
		ui.selGeometryTable->item( 1, 0 )->setText( QString::number( pGeom->vTriangles.size() ) );
		const SVolumeType bbox = pGeom->BoundingBox();
		ui.selGeometryTable->item(2, 0)->setText(QString::number(bbox.coordEnd.x - bbox.coordBeg.x));
		ui.selGeometryTable->item(3, 0)->setText(QString::number(bbox.coordEnd.y - bbox.coordBeg.y));
		ui.selGeometryTable->item(4, 0)->setText(QString::number(bbox.coordEnd.z - bbox.coordBeg.z));
	}
	else
		for ( int i = 0; i < ui.selGeometryTable->rowCount(); i++ )
			ui.selGeometryTable->item( i, 0 )->setText( "" );
}


void CGeometriesDatabaseTab::NewGeometryAdded()
{
	emit UpdateOpenGLView();
	emit GeometryAdded();
}


void CGeometriesDatabaseTab::UpdateButtons()
{
	if (!m_pGeometriesDB) return;
	QString sLockerFileName = ss2qs(m_pGeometriesDB->GetFileName()) + ".lock";
	QLockFile *newFileLocker = new QLockFile(sLockerFileName);
	newFileLocker->setStaleLockTime(0);
	bool bSuccessfullyLocked = newFileLocker->tryLock(10);
	newFileLocker->unlock();
	ui.saveDatabase->setEnabled(bSuccessfullyLocked);
}
