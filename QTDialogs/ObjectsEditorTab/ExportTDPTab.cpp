/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ExportTDPTab.h"
#include "qtOperations.h"
#include <QFileDialog>
#include <QMessageBox>

CExportTDPTab::CExportTDPTab( QWidget *parent )	:CMusenDialog( parent )
{
	ui.setupUi(this);

	setWindowModality(Qt::ApplicationModal);
	m_sOutputFileName = ".\\ExprotedData.csv";
	m_dMinTime = 0;
	m_dTimeStep = 0;
	m_dMaxTime = 0;
	InitializeConnections();
}


void CExportTDPTab::setVisible( bool _bVisible )
{
	if (( _bVisible == true ) && ( m_dMaxTime == -1 ))
		m_dMaxTime = m_pSystemStructure->GetMaxTime();
	ui.statusLabel->setText( "     " );
	UpdateWholeView();
	QDialog::setVisible( _bVisible );
}


void CExportTDPTab::UpdateWholeView()
{
	m_bAvoidSignal = true;

	if ( m_dMaxTime > m_pSystemStructure->GetMaxTime() )
		m_dMaxTime = m_pSystemStructure->GetMaxTime();
	if ( m_dMinTime >= m_dMaxTime )
		m_dMinTime = m_dMaxTime;
	if ( m_dTimeStep <= 0 )
		m_dTimeStep = (m_dMaxTime - m_dMinTime)/1000.0;

	ui.selObjectsNumber->setText( QString::number( m_vSelectedObjectsID.size() ) );
	ui.minTime->setText( QString::number( m_dMinTime ) );
	ui.maxTimeValue->setText( " max=" + QString::number( m_pSystemStructure->GetMaxTime() ) + "[s]");
	ui.maxTime->setText( QString::number( m_dMaxTime ) );
	ui.timeStep->setText( QString::number( m_dTimeStep ) );
	ui.outputFileName->setText( m_sOutputFileName );

	m_bAvoidSignal = false;
}


void CExportTDPTab::InitializeConnections()
{
	QObject::connect( ui.minTime, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
	QObject::connect( ui.maxTime, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
	QObject::connect( ui.timeStep, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );

	QObject::connect( ui.selectFile, SIGNAL(clicked()), this, SLOT(ChangeFileName()) );
	QObject::connect( ui.exportButton, SIGNAL(clicked()), this, SLOT(ExportData()) );
}


void CExportTDPTab::DataWasChanged()
{
	if ( m_bAvoidSignal ) return;
	m_dMaxTime = ui.maxTime->text().toDouble();
	m_dMinTime = ui.minTime->text().toDouble();
	m_dTimeStep = ui.timeStep->text().toDouble();
}


void CExportTDPTab::ChangeFileName()
{
	QString sFileName = QFileDialog::getSaveFileName(this, tr("Export data"), "", tr( "CSV files (*.csv);;All files (*.*);;" ));
	if ( sFileName.simplified() != "" )
		m_sOutputFileName = sFileName;
	UpdateWholeView();
}


void CExportTDPTab::ExportData()
{
	if (!IsFileWritable(m_sOutputFileName))
	{
		QMessageBox::warning(this, "Writing error", "Unable to export - selected file is not writable");
		return;
	}

	// open file for writing
	std::ofstream outputFile;
	outputFile.open(UnicodePath(qs2ss( m_sOutputFileName )) );
	if ( outputFile.fail() )
		return;

	ui.statusLabel->setText( "Exporting started" );
	ui.exportButton->setEnabled( false );
	ui.cancelButton->setEnabled( false );
	ui.selectFile->setEnabled( false );
	ui.dataSeparator->setEnabled( false );

	std::string sSeparator = qs2ss( ui.dataSeparator->text() );
	bool bExportCoord = ui.exprotCoordinates->isChecked();
	bool bExportVel = ui.exportVelocities->isChecked();
	bool bExportForces = ui.exportForces->isChecked();
	bool bExportAnglVel = ui.exportAngularVelocities->isChecked();

	// go through all time points
	double dCurrentTime = m_dMinTime;
	while ( dCurrentTime <= m_dMaxTime )
	{
		if ( dCurrentTime == m_dMinTime ) // create header if this is first time point
		{
			outputFile << "Time[s]" << sSeparator;
			for ( unsigned i=0; i < m_vSelectedObjectsID.size(); i++ )
			{
				if ( bExportCoord )
				{
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_X" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Y" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Z" << sSeparator;
				}
				if ( bExportVel )
				{
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Vx" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Vy" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Vz" << sSeparator;
				}
				if ( bExportAnglVel )
				{
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Wx" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Wy" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Wz" << sSeparator;
				}
				if ( bExportForces )
				{
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Force_x" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Force_y" << sSeparator;
					outputFile << "ID_" << m_vSelectedObjectsID[ i ] << "_Force_z" << sSeparator;
				}
			}
			outputFile << std::endl;
		}

		outputFile << dCurrentTime << sSeparator;
		// export information about objects
		for ( unsigned i=0; i < m_vSelectedObjectsID.size(); i++ )
		{
			CPhysicalObject* pTemp = m_pSystemStructure->GetObjectByIndex( m_vSelectedObjectsID[ i ] );
			if ( pTemp == NULL ) continue;
			CVector3 tempVector;
			if ( bExportCoord )
			{
				tempVector = pTemp->GetCoordinates( dCurrentTime );
				outputFile << tempVector.x << sSeparator << tempVector.y << sSeparator << tempVector.z << sSeparator;
			}
			if ( bExportVel )
			{
				tempVector = pTemp->GetVelocity( dCurrentTime );
				outputFile << tempVector.x << sSeparator << tempVector.y << sSeparator << tempVector.z << sSeparator;
			}
			if ( bExportAnglVel )
			{
				tempVector = pTemp->GetAngleVelocity( dCurrentTime );
				outputFile << tempVector.x << sSeparator << tempVector.y << sSeparator << tempVector.z << sSeparator;
			}
			if ( bExportForces )
			{
				tempVector = pTemp->GetForce( dCurrentTime );
				outputFile << tempVector.x << sSeparator << tempVector.y << sSeparator << tempVector.z << sSeparator;
			}
		}
		outputFile << std::endl;
		dCurrentTime += m_dTimeStep;
	}
	outputFile.close();

	ui.statusLabel->setText( "Exporting finished" );
	ui.exportButton->setEnabled( true );
	ui.cancelButton->setEnabled( true );
	ui.selectFile->setEnabled( true );
	ui.dataSeparator->setEnabled( true );
}