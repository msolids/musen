/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AgglomeratesDatabaseTab.h"

CAgglomeratesDatabaseTab::CAgglomeratesDatabaseTab(QWidget *parent )
	: CMusenDialog( parent )
{
	ui.setupUi(this);
	m_pInsertAgglomTab = new CInsertAgglomTab( this );

	ui.selAgglomerateTable->horizontalHeader()->setSectionResizeMode( QHeaderView::ResizeMode::Stretch );
	ui.selAgglomerateTable->verticalHeader()->setSectionResizeMode( QHeaderView::ResizeMode::Stretch );


	m_sHelpFileName = "Users Guide/Agglomerates Database.pdf";
	InitializeConnections();
	for ( int i = 0; i<ui.selAgglomerateTable->rowCount(); i++ )
		ui.selAgglomerateTable->item( i, 0 )->setFlags( ui.selAgglomerateTable->item( i, 0 )->flags() & (~Qt::ItemIsEditable) );
}

void CAgglomeratesDatabaseTab::InitializeConnections()
{
	connect( ui.addAgglomerate, SIGNAL( clicked() ), this, SLOT( AddAgglomerate() ) );
	connect( ui.newDatabase, SIGNAL( clicked() ), this, SLOT( NewDatabase() ) );
	connect( ui.saveDatabase, SIGNAL(clicked()), this, SLOT(SaveDatabase()) );
	connect( ui.saveDatabaseAs, SIGNAL( clicked() ), this, SLOT( SaveDatabaseAs() ) );
	connect( ui.loadDatabase, SIGNAL( clicked() ), this, SLOT( LoadDatabase() ) );
	connect( ui.deleteAgglom, SIGNAL( clicked() ), this, SLOT( DeleteAgglomerate() ) );
	connect( ui.upAgglomerate, SIGNAL(clicked()), this, SLOT(UpAgglomerate()));
	connect( ui.downAgglomerate, SIGNAL(clicked()), this, SLOT(DownAgglomerate()));


	connect( ui.agglomeratesList, SIGNAL( currentItemChanged( QListWidgetItem*, QListWidgetItem* ) ), this, SLOT( NewRowSelected() ) );
	connect( ui.agglomeratesList, SIGNAL( itemChanged( QListWidgetItem* ) ), this, SLOT( DataWasChanged() ) );

	connect( ui.insertAgglomerate, SIGNAL(clicked()), this, SLOT( InsertAgglomerate() ) );
	connect( m_pInsertAgglomTab, SIGNAL( NewAgglomerateAdded() ), this, SLOT( NewAgglomerateAdded() ) );
}

void CAgglomeratesDatabaseTab::DeleteAgglomerate()
{
	m_pAgglomDB->DeleteAgglomerate( ui.agglomeratesList->currentRow() );
	UpdateWholeView();
}

void CAgglomeratesDatabaseTab::UpAgglomerate()
{
	int oldRow = ui.agglomeratesList->currentRow();
	m_pAgglomDB->UpAgglomerate(oldRow);
	UpdateWholeView();
	ui.agglomeratesList->setCurrentRow( oldRow == 0 ? 0 : --oldRow );
}

void CAgglomeratesDatabaseTab::DownAgglomerate()
{
	int oldRow = ui.agglomeratesList->currentRow();
	int lastRow = ui.agglomeratesList->count() - 1;
	m_pAgglomDB->DownAgglomerate(oldRow);
	UpdateWholeView();
	ui.agglomeratesList->setCurrentRow( oldRow == lastRow ? lastRow : ++oldRow );
}

void CAgglomeratesDatabaseTab::AddAgglomerate()
{
	// add current system structure as agglomerate
	SAgglomerate* pNewAgglomerate = new SAgglomerate;
	pNewAgglomerate->sKey = GenerateKey();
	pNewAgglomerate->sName = "New agglomerate";
	std::vector<unsigned> newSpheresIndexes( m_pSystemStructure->GetTotalObjectsCount() ); // new indexes of spheres in the new array
	QSet<QString> partCompounds, bondCompounds;
	for ( unsigned i = 0; i<m_pSystemStructure->GetTotalObjectsCount(); i++ )
	{
		CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex( i );
		if ( pObject == NULL ) continue;
		if ( pObject->GetObjectType() == SPHERE )
		{
			newSpheresIndexes[i] = (unsigned)pNewAgglomerate->vParticles.size();
			std::string sCompound = pObject->GetCompoundKey();
			pNewAgglomerate->vParticles.push_back(SAggloParticle(pObject->GetCoordinates(0), pObject->GetOrientation(0), ((CSphere*)pObject)->GetRadius(), ((CSphere*)pObject)->GetContactRadius(), sCompound));
			partCompounds.insert(ss2qs(pObject->GetCompoundKey()));
		}
	}

	for ( unsigned i = 0; i<m_pSystemStructure->GetTotalObjectsCount(); i++ )
	{
		CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex( i );
		if ( pObject == NULL ) continue;
		if ( pObject->GetObjectType() == SOLID_BOND )
		{
			std::string sCompound = pObject->GetCompoundKey();
			pNewAgglomerate->vBonds.push_back( SAggloBond( ((CSolidBond*)pObject)->GetDiameter() / 2,
				newSpheresIndexes[ ((CSolidBond*)pObject)->m_nLeftObjectID ],
				newSpheresIndexes[ ((CSolidBond*)pObject)->m_nRightObjectID ], sCompound ) );
			bondCompounds.insert(ss2qs(pObject->GetCompoundKey()));
		}
	}

	CAgglomCompounds dialog(pNewAgglomerate, m_pMaterialsDB, partCompounds.values(), bondCompounds.values());
	if (dialog.exec() != QDialog::Accepted)
		return;
	m_pAgglomDB->AddNewAgglomerate( *pNewAgglomerate );
	UpdateWholeView();
}

void CAgglomeratesDatabaseTab::ShowAgglomerate()
{
	int nIndex = ui.agglomeratesList->currentRow();
	if (nIndex == -1) return;
	SAgglomerate* pAgglom = m_pAgglomDB->GetAgglomerate(nIndex);
	if (!pAgglom) return;
	ui.agglomOpenGLView->SetCurrentAgglomerate(pAgglom);
}

void CAgglomeratesDatabaseTab::NewDatabase()
{
	m_pAgglomDB->NewDatabase();
	UpdateWholeView();
}

void CAgglomeratesDatabaseTab::DataWasChanged()
{
	if (m_bAvoidSignal) return;
	for ( int i = 0; i < ui.agglomeratesList->count(); ++i )
	{
		SAgglomerate* pAgglom = m_pAgglomDB->GetAgglomerate(i);
		if (!pAgglom) continue;
		pAgglom->sName = qs2ss( ui.agglomeratesList->item( i )->text() );
	}
	UpdateWholeView();
}

void CAgglomeratesDatabaseTab::LoadDatabase()
{
	QString sFileName = QFileDialog::getOpenFileName( this, tr( "Load database" ), "", tr( "Process (*.madb);;All files (*.*);;" ) );
	if ( sFileName.simplified() == "" )
		return;

	bool bIsOldFileFromat = m_pAgglomDB->LoadFromFile(qs2ss( sFileName ));
	if (bIsOldFileFromat)
	{
		QMessageBox::warning(this, "Load agglomerate database", "Selected file has the old format. It will be transformed into the new format.");
		m_pAgglomDB->SaveToFile(qs2ss(sFileName));
	}
	UpdateWholeView();
}

void CAgglomeratesDatabaseTab::SaveDatabase()
{
	SaveDatabaseAs( ss2qs( m_pAgglomDB->GetFileName() ) );
}

void CAgglomeratesDatabaseTab::SaveDatabaseAs( const QString& _sFileName /*= "" */ )
{
	QString sFileName = _sFileName;
	if ( sFileName.simplified() == "" )
		sFileName = QFileDialog::getSaveFileName(this, tr("Save database"), "", tr( "Processes (*.madb);;All files (*.*);;" ));
	if ( sFileName.simplified() == "" )
		return;

	m_pAgglomDB->SaveToFile( qs2ss( sFileName ) );
	UpdateWholeView();
}

void CAgglomeratesDatabaseTab::UpdateWholeView()
{
	m_bAvoidSignal = true;
	ui.agglomeratesList->clear();
	setWindowTitle( "Agglomerates\\Multisphere database: " + ss2qs( m_pAgglomDB->GetFileName() ) );
	UpdateButtons();

	for (unsigned i = 0; i < m_pAgglomDB->GetAgglomNumber(); ++i)
	{
		SAgglomerate* pAgglom = m_pAgglomDB->GetAgglomerate(i);
		if (!pAgglom) continue;
		ui.agglomeratesList->insertItem( i, new QListWidgetItem( ss2qs( pAgglom->sName ) ) );
		ui.agglomeratesList->item( i )->setFlags( ui.agglomeratesList->item( i )->flags() | Qt::ItemIsEditable );
	}
	m_bAvoidSignal = false;
	if ( (ui.agglomeratesList->currentRow() == -1) && (m_pAgglomDB->GetAgglomNumber() > 0) )
		ui.agglomeratesList->setCurrentRow( 0 );
	NewRowSelected();
	UpdateSelectedAgglomInfo();
}

void CAgglomeratesDatabaseTab::NewRowSelected()
{
	int nCurrentRow = ui.agglomeratesList->currentRow();
	SAgglomerate* pAgglom = nullptr;
	if ( nCurrentRow >= 0 )
		pAgglom = m_pAgglomDB->GetAgglomerate(nCurrentRow);
	ui.deleteAgglom->setEnabled( pAgglom != NULL );
	ui.insertAgglomerate->setEnabled(m_bEnableInsertion && (pAgglom != nullptr));
	if (isVisible())
		ShowAgglomerate();
	UpdateSelectedAgglomInfo();
}

void CAgglomeratesDatabaseTab::UpdateSelectedAgglomInfo()
{
	int nCurrentRow = ui.agglomeratesList->currentRow();
	SAgglomerate* pAgglom = NULL;
	if ( nCurrentRow >= 0 )
		pAgglom = m_pAgglomDB->GetAgglomerate( nCurrentRow );

	ShowConvLabel( ui.selAgglomerateTable->verticalHeaderItem( 4 ), "Volume", EUnitType::VOLUME );
	ui.selAgglomerateTable->setEnabled( pAgglom != NULL );
	if ( pAgglom != NULL )
	{
		ui.selAgglomerateTable->item( 0, 0 )->setText( ss2qs( pAgglom->sName ) );
		if ( pAgglom->nType == AGGLOMERATE )
			ui.selAgglomerateTable->item( 1, 0 )->setText( "Bonded agglomerate" );
		else
			ui.selAgglomerateTable->item( 1, 0 )->setText( "Multisphere" );
		ui.selAgglomerateTable->item( 2, 0 )->setText( QString::number( pAgglom->vParticles.size() ) );
		ui.selAgglomerateTable->item( 3, 0 )->setText( QString::number( pAgglom->vBonds.size() ));
		ShowConvValue( ui.selAgglomerateTable->item( 4, 0 ), pAgglom->dVolume, EUnitType::VOLUME );
	}
	else
		for ( int i = 0; i < ui.selAgglomerateTable->rowCount(); ++i )
			ui.selAgglomerateTable->item( i, 0 )->setText( "" );
}


void CAgglomeratesDatabaseTab::InsertAgglomerate()
{
	int nCurrentRow = ui.agglomeratesList->currentRow();
	if (nCurrentRow == -1) return;
	SAgglomerate* pAgglom = m_pAgglomDB->GetAgglomerate(nCurrentRow);
	if (!pAgglom) return;
	m_pInsertAgglomTab->SetCurrentAgglom(pAgglom->sKey);
	m_pInsertAgglomTab->show();
	m_pInsertAgglomTab->raise();
}

void CAgglomeratesDatabaseTab::NewAgglomerateAdded()
{
	emit UpdateOpenGLView();
	emit AgglomerateAdded();
}

void CAgglomeratesDatabaseTab::SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB)
{
	CMusenDialog::SetPointers(_pSystemStructure, _pUnitConvertor, _pMaterialsDB, _pGeometriesDB, _pAgglomDB);
	m_pInsertAgglomTab->SetPointers(_pSystemStructure, _pUnitConvertor, _pMaterialsDB, _pGeometriesDB, _pAgglomDB);
}

void CAgglomeratesDatabaseTab::EnableInsertion(bool _bEnable)
{
	m_bEnableInsertion = _bEnable;
	NewRowSelected();
}

void CAgglomeratesDatabaseTab::UpdateButtons()
{
	if (!m_pAgglomDB) return;
	QString sLockerFileName = ss2qs(m_pAgglomDB->GetFileName()) + ".lock";
	QLockFile *newFileLocker = new QLockFile(sLockerFileName);
	newFileLocker->setStaleLockTime(0);
	bool bSuccessfullyLocked = newFileLocker->tryLock(10);
	newFileLocker->unlock();
	ui.saveDatabase->setEnabled(bSuccessfullyLocked);
}

