/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SceneInfoTab.h"
#include <QFileDialog>
#include <QMessageBox>

CSceneInfoTab::CSceneInfoTab( QWidget *parent ) : CMusenDialog( parent )
{
	m_dCurrentTime = 0;
	ui.setupUi(this);

	ui.infoTable->verticalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.infoTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	InitializeConnections();
}


void CSceneInfoTab::InitializeConnections()
{
	// signals from buttons
	QObject::connect( ui.updateInfoButton, SIGNAL(clicked()), this, SLOT(UpdateInfo()) );
	QObject::connect(ui.updateDeatiledInfoButton, SIGNAL(clicked()), this, SLOT(UpdateDetailedInfo()));
	QObject::connect( ui.exportOptions, SIGNAL(clicked()), &m_ExportOptionsTab, SLOT(show()) );
	QObject::connect( ui.exportData, SIGNAL(clicked()), this, SLOT(ExportResults()) );

	// signal from combobox
	QObject::connect( ui.selectedMaterialCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(UpdateTableView()) );
	m_ExportOptionsTab.InitializeConnections();
}


void CSceneInfoTab::UpdateComboBoxList()
{
	 //update information in the combobox
	int nOldSelectedIndex = ui.selectedMaterialCombo->currentIndex();
	ui.selectedMaterialCombo->clear();
	ui.selectedMaterialCombo->insertItem( 0, "All" );
	for ( unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); i++ )
		ui.selectedMaterialCombo->insertItem( i+1, ss2qs( m_pMaterialsDB->GetCompoundName(i) ) );
	if (nOldSelectedIndex <= (int)m_pMaterialsDB->CompoundsNumber())
		ui.selectedMaterialCombo->setCurrentIndex( nOldSelectedIndex );
	else
		ui.selectedMaterialCombo->setCurrentIndex( 0 );
}


void CSceneInfoTab::UpdateTableView()
{
	if ( ui.selectedMaterialCombo->currentIndex() < 0 )
		return;
	QString sSelectedMaterial;
	if ( ui.selectedMaterialCombo->currentIndex() != 0 ) // if not all
		sSelectedMaterial = ss2qs( m_pMaterialsDB->GetCompoundKey( ui.selectedMaterialCombo->currentIndex()-1 ) );

	// particles number/mass/volume
	std::vector<CPhysicalObject*> vecMaterialParticles, vecAllParticles;
	m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( m_dCurrentTime, &vecMaterialParticles, SPHERE, qs2ss( sSelectedMaterial ) );
	m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( m_dCurrentTime, &vecAllParticles, SPHERE );
	unsigned nMaterialParticlesNumber = (unsigned)vecMaterialParticles.size();
	unsigned nTotalParticlesNumber = (unsigned)vecAllParticles.size();
	double dSelectedMass = 0; double dSelectedVolume = 0;
	double dTotalMass = 0; double dTotalVolume = 0;
	for ( size_t i=0; i < vecAllParticles.size(); i++ )
	{
		dTotalMass += vecAllParticles[ i ]->GetMass();
		dTotalVolume += ((CSphere*)vecAllParticles[ i ])->GetVolume();
	}

	for ( unsigned i=0; i < vecMaterialParticles.size(); i++ )
	{
		dSelectedMass += vecMaterialParticles[ i ]->GetMass();
		dSelectedVolume += ((CSphere*)vecMaterialParticles[ i ])->GetVolume();
	}

	ui.infoTable->item( 0, 0 )->setText( QString::number( nMaterialParticlesNumber ) );
	ShowConvValue( ui.infoTable->item( 0, 1 ), dSelectedMass, EUnitType::MASS );
	ShowConvValue( ui.infoTable->item( 0, 2 ), dSelectedVolume, EUnitType::VOLUME );
	if ( nTotalParticlesNumber > 0 )
		ui.infoTable->item( 1, 0 )->setText( QString::number( 100*nMaterialParticlesNumber/(double)nTotalParticlesNumber ) );
	else
		ui.infoTable->item( 1, 0 )->setText( "100" );
	if ( dTotalMass > 0 )
		ui.infoTable->item( 1, 1 )->setText( QString::number( 100*dSelectedMass/dTotalMass ) );
	else
		ui.infoTable->item( 1, 1 )->setText( "100" );
	if ( dTotalVolume > 0 )
		ui.infoTable->item( 1, 2 )->setText( QString::number( 100*dSelectedVolume/dTotalVolume ) );
	else
		ui.infoTable->item( 1, 2 )->setText( "100" );



	// solid bonds number/mass/volume
	std::vector<CPhysicalObject*> vecMaterialSolidBonds, vecAllSolidBonds;
	m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( m_dCurrentTime, &vecMaterialSolidBonds, SOLID_BOND, qs2ss( sSelectedMaterial ) );
	m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( m_dCurrentTime, &vecAllSolidBonds, SOLID_BOND );
	unsigned nMaterialSolidBondsNumber = (unsigned)vecMaterialSolidBonds.size();
	unsigned nTotalSolidBondsNumber = (unsigned)vecAllSolidBonds.size();
	dSelectedVolume = 0;
	dTotalVolume = 0;
	for ( unsigned i=0; i < vecAllSolidBonds.size(); i++ )
		dTotalVolume += m_pSystemStructure->GetBondVolume( m_dCurrentTime, vecAllSolidBonds[ i ]->m_lObjectID );

	for ( unsigned i=0; i < vecMaterialSolidBonds.size(); i++ )
		dSelectedVolume += m_pSystemStructure->GetBondVolume( m_dCurrentTime, vecMaterialSolidBonds[ i ]->m_lObjectID );

	ui.infoTable->item( 2, 0 )->setText( QString::number( nMaterialSolidBondsNumber ) );
	ui.infoTable->item( 2, 1 )->setText( "0" );
	ShowConvValue( ui.infoTable->item( 2, 2 ), dSelectedVolume, EUnitType::VOLUME );
	if ( nTotalSolidBondsNumber > 0 )
		ui.infoTable->item( 3, 0 )->setText( QString::number( 100*nMaterialSolidBondsNumber/(double)nTotalSolidBondsNumber ) );
	else
		ui.infoTable->item( 3, 0 )->setText( "100" );
	ui.infoTable->item( 3, 1 )->setText( "0" );
	if ( dTotalVolume > 0 )
		ui.infoTable->item( 3, 2 )->setText( QString::number( 100*dSelectedVolume/dTotalVolume ) );
	else
		ui.infoTable->item( 3, 2 )->setText( "100" );




	// liquid bonds number/mass/volume
	std::vector<CPhysicalObject*> vecMaterialWalls, vecAllWalls;
	m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( m_dCurrentTime, &vecMaterialWalls, TRIANGULAR_WALL, qs2ss( sSelectedMaterial ) );
	m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( m_dCurrentTime, &vecAllWalls, TRIANGULAR_WALL);
	unsigned nMaterialWallsNumber = (unsigned)vecMaterialWalls.size();
	unsigned nTotalWallsNumber = (unsigned)vecAllWalls.size();
	dSelectedVolume = 0;
	dTotalVolume = 0;

	ui.infoTable->item( 4, 0 )->setText( QString::number( nMaterialWallsNumber ) );
	ui.infoTable->item( 4, 1 )->setText( "0" );
	ShowConvValue( ui.infoTable->item( 4, 2 ), dSelectedVolume, EUnitType::VOLUME );
	if ( nTotalWallsNumber > 0 )
		ui.infoTable->item( 5, 0 )->setText( QString::number( 100*nMaterialWallsNumber/(double)nTotalWallsNumber ) );
	else
		ui.infoTable->item( 5, 0 )->setText( "100" );
	ui.infoTable->item( 5, 1 )->setText( "0" );
	if ( dTotalVolume > 0 )
		ui.infoTable->item( 5, 2 )->setText( QString::number( 100*dSelectedVolume/dTotalVolume ) );
	else
		ui.infoTable->item( 5, 2 )->setText( "100" );
}


void CSceneInfoTab::UpdateMaxOverlap()
{
	if ( this->isVisible() == false )
		return;

	double dMaxOverlap, dTotalOverlap, dAverageOverlap, dAverageContactRadius;
	dMaxOverlap = dTotalOverlap = dAverageOverlap = dAverageContactRadius = 0;
	std::vector<unsigned> vID1, vID2;
	unsigned nID1, nID2;
	std::vector<double> vOverlaps;
	m_pSystemStructure->GetOverlaps( m_dCurrentTime, vID1, vID2, vOverlaps );
	for ( unsigned i = 0; i < vOverlaps.size(); ++i )
	{
		dTotalOverlap += vOverlaps[ i ];
		if ( vOverlaps[ i ] > dMaxOverlap )
		{
			dMaxOverlap = vOverlaps[ i ];
			nID1 = vID1[ i ];
			nID2 = vID2[ i ];
		}
		const auto part1 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(vID1[i]));
		const auto part2 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(vID2[i]));
		const double dR1 = part1 ? part1->GetRadius() : 0;
		const double dR2 = part2 ? part2->GetRadius() : 0;
		const double dSpecEquivRadius = dR1 + dR2 != 0 ? 2 * dR1*dR2 / (dR1 + dR2) : 0;
		dAverageContactRadius += sqrt( 2*dSpecEquivRadius*vOverlaps[ i ] );
	}

	/*Average contact radius is calculated according to publication Besler et al. 2015.*/

	if ( vOverlaps.size() > 0 )
	{
		dAverageOverlap = dTotalOverlap/vOverlaps.size();
		dAverageContactRadius = dAverageContactRadius/vOverlaps.size();
	}

	//m_pSystemStructure->GetOverlaps( dMaxOverlap, nID1, nID2, dTotalOverlap, m_dCurrentTime );
	ui.maxOverlap->setText( QString::number( m_pUnitConverter->GetValue(EUnitType::PARTICLE_DIAMETER, dMaxOverlap) )
		+ " ID1:" + QString::number( nID1 ) + " ID2:" + QString::number( nID2 ) );
	ShowConvValue( ui.averageOverlap, dAverageOverlap, EUnitType::PARTICLE_DIAMETER );
	ShowConvValue( ui.totalOverlap, dTotalOverlap, EUnitType::PARTICLE_DIAMETER );
	ShowConvValue( ui.averageContactRadius, dAverageContactRadius, EUnitType::PARTICLE_DIAMETER );

	ui.contactsNumber->setText( QString::number( vOverlaps.size() ) );
}


void CSceneInfoTab::UpdateLabelsHeaders()
{
	ShowConvLabel( ui.timeLabel, "Current time", EUnitType::TIME );
	ShowConvLabel( ui.totalOverlapLabel,  "Total overlap", EUnitType::PARTICLE_DIAMETER );
	ShowConvLabel( ui.maxOverlapLabel, "Max overlap", EUnitType::PARTICLE_DIAMETER );
	ShowConvLabel( ui.averageOverlapLabel, "Average overlap", EUnitType::PARTICLE_DIAMETER );
	ShowConvLabel( ui.averageContactRadiusLabel, "Average contact radius", EUnitType::PARTICLE_DIAMETER );
	ShowConvLabel( ui.infoTable->horizontalHeaderItem( 1 ), "Mass", EUnitType::MASS );
	ShowConvLabel( ui.infoTable->horizontalHeaderItem( 2 ), "Volume", EUnitType::VOLUME );
}

void CSceneInfoTab::UpdateInfo()
{
	ShowConvValue(ui.currentTime, m_dCurrentTime, EUnitType::TIME);
	UpdateLabelsHeaders();
	UpdateComboBoxList();
	UpdateTableView();
	m_ExportOptionsTab.UpdateWholeView();
}

void CSceneInfoTab::UpdateDetailedInfo()
{
	UpdateMaxOverlap();
	UpdateAgglomNumber();
}

void CSceneInfoTab::UpdateAgglomNumber()
{
	CAgglomeratesAnalyzer agglomAnalyzer;
	agglomAnalyzer.SetSystemStructure( m_pSystemStructure );
	agglomAnalyzer.FindAgglomerates( m_dCurrentTime );
	ui.agglomeratesNumber->setText( QString::number( agglomAnalyzer.GetAgglomeratesNumber() ) );
}

void CSceneInfoTab::closeEvent( QCloseEvent * event )
{
	m_ExportOptionsTab.close();
	event->accept();
}


void CSceneInfoTab::ExportResults()
{
	QString sFileName = QFileDialog::getSaveFileName(this, tr("Export results"), "", tr( "Results data (*.csv);;All files (*.*);;" ));
	if (sFileName.simplified().isEmpty())
		return;
	if (!IsFileWritable(sFileName))
	{
		QMessageBox::warning(this, "Writing error", "Unable to export - selected file is not writable");
		return;
	}

	std::ofstream outFile;
	outFile.open(UnicodePath(qs2ss(sFileName))); //open a file

	// separator
	std::string sSep = qs2ss( m_ExportOptionsTab.m_sSeparator );

	// export PSD
	double dMinDiameter = m_pSystemStructure->GetMinParticleDiameter();
	double dMaxDiameter = m_pSystemStructure->GetMaxParticleDiameter();
	double dClassSize = (dMaxDiameter-dMinDiameter)/m_ExportOptionsTab.m_nNumberOfPSDClasses;
	std::vector<unsigned> vecPSD = m_pSystemStructure->GetPrimaryPSD( 0, dMinDiameter, dMaxDiameter+dClassSize, m_ExportOptionsTab.m_nNumberOfPSDClasses );
	outFile<<"Number distribution of primary particles" << std::endl;
	outFile<<"Lower boundary diameter (included) [m]" << sSep;
	for ( unsigned i=0; i < vecPSD.size(); i++ )
		outFile<< dMinDiameter + i*(dMaxDiameter-dMinDiameter)/vecPSD.size() << sSep;
	outFile<<std::endl;
	outFile<<"Upper boundary diameter (excluded)[m]" << sSep;
	for ( unsigned i=0; i < vecPSD.size(); i++ )
		outFile<< dMinDiameter + i*(dMaxDiameter-dMinDiameter)/vecPSD.size() + dClassSize << sSep;
	outFile<<std::endl;
	outFile<<"Particles number" << sSep;
	for ( unsigned i=0; i < vecPSD.size(); i++ )
		outFile<< vecPSD[ i ] << sSep;
	outFile<<std::endl;

	outFile<<std::endl<<std::endl;

	// time dependent properties
	outFile<<"Time" << sSep << "Solid bonds" << sSep << "Max overlap[m]" << sSep << "Total overlap[m]" << std::endl;

	int nStepsNumber = (m_ExportOptionsTab.m_dEndTime-m_ExportOptionsTab.m_dStartTime)/m_ExportOptionsTab.m_dTimeStep;
	if ( nStepsNumber <= 0 ) nStepsNumber = 0;
	for ( int i=0; i < nStepsNumber; i++ )
	{
		double dTime = m_ExportOptionsTab.m_dTimeStep*i;
		double dMaxOverlap, dTotalOverlap;
		unsigned nID1, nID2;
		m_pSystemStructure->GetOverlaps( dMaxOverlap, nID1, nID2, dTotalOverlap, dTime );

		std::vector<CPhysicalObject*> vecAllSolidBonds;
		m_pSystemStructure->GetAllObjectsOfSpecifiedCompound( dTime, &vecAllSolidBonds, SOLID_BOND );
		outFile << m_ExportOptionsTab.m_dTimeStep*i << sSep << vecAllSolidBonds.size() << sSep << dMaxOverlap << sSep << dTotalOverlap << std::endl;
	}

	outFile.close();
}
