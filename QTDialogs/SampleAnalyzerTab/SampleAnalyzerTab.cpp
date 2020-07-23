/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SampleAnalyzerTab.h"
#include "qtOperations.h"
#include <QFileDialog>

CSampleAnalyzerTab::CSampleAnalyzerTab( QWidget *parent ): CMusenDialog( parent )
{
	m_dCurrentTime = 0;
	m_dRadius = 1e-3;
	m_vCenter.Init(0);
	ui.setupUi(this);

	// default export options
	m_dPSDStart = 0;
	m_dPSDEnd = 1e-3;
	m_nPSDIntervalsNumber = 10;
	m_dVelStart = 0;
	m_dVelEnd = 1;
	m_nVelIntervalsNumber = 10;
	m_dTimeStart = 0;
	m_dTimeEnd = 0.1;
	m_nTimeIntervalsNumber = 10;

	InitializeConnections();
}


void CSampleAnalyzerTab::InitializeConnections()
{
	// signals from volume group box
	QObject::connect( ui.xCoord, SIGNAL(editingFinished()), this, SLOT(VolumeDataWasChanged()) );
	QObject::connect( ui.yCoord, SIGNAL(editingFinished()), this, SLOT(VolumeDataWasChanged()) );
	QObject::connect( ui.zCoord, SIGNAL(editingFinished()), this, SLOT(VolumeDataWasChanged()) );
	QObject::connect( ui.radius, SIGNAL(editingFinished()), this, SLOT(VolumeDataWasChanged()) );

	// signals from export data
	QObject::connect( ui.exportPSD, SIGNAL(clicked()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.exportVel, SIGNAL(clicked()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.timeStart, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.timeEnd, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.timeIntervalsNumber, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.psdStart, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.psdEnd, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.psdIntervalsNumber, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.velStart, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.velEnd, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );
	QObject::connect( ui.velIntervalsNumber, SIGNAL(editingFinished()), this, SLOT(ExportDataWasChanged()) );

	// from update button
	QObject::connect( ui.updateMainData, SIGNAL(clicked()), this, SLOT(UpdateMainProperties()) );

	//from export button
	QObject::connect( ui.exportData, SIGNAL(clicked()), this, SLOT(ExportData()) );
}


void CSampleAnalyzerTab::UpdateVolumeView()
{
	m_bAvoidSignal = true;
	ShowConvLabel( ui.xLabel, "X", EUnitType::LENGTH );	ShowConvLabel( ui.yLabel, "Y", EUnitType::LENGTH );	ShowConvLabel( ui.zLabel, "Z", EUnitType::LENGTH );
	ShowConvLabel( ui.radiusLabel, "Radius", EUnitType::LENGTH );
	ShowConvValue( ui.xCoord, m_vCenter.x, EUnitType::LENGTH );
	ShowConvValue( ui.yCoord, m_vCenter.y, EUnitType::LENGTH );
	ShowConvValue( ui.zCoord, m_vCenter.z, EUnitType::LENGTH );
	ShowConvValue( ui.radius, m_dRadius, EUnitType::LENGTH );
	m_bAvoidSignal = false;
}


void CSampleAnalyzerTab::UpdateExportView()
{
	m_bAvoidSignal = true;
	ShowConvLabel( ui.psdLabel, "PSD", EUnitType::PARTICLE_DIAMETER );
	ShowConvLabel( ui.velocityLabel, "Velocity", EUnitType::VELOCITY );

	// set parameters for PSD
	ui.psdStart->setEnabled( ui.exportPSD->isChecked() );
	ui.psdEnd->setEnabled( ui.exportPSD->isChecked() );
	ui.psdIntervalsNumber->setEnabled( ui.exportPSD->isChecked() );
	ShowConvValue( ui.psdStart, m_dPSDStart, EUnitType::PARTICLE_DIAMETER );
	ShowConvValue( ui.psdEnd, m_dPSDEnd, EUnitType::PARTICLE_DIAMETER );
	ui.psdIntervalsNumber->setText( QString::number( m_nPSDIntervalsNumber ) );

	// set parameters for velocities
	ui.velStart->setEnabled( ui.exportVel->isChecked() );
	ui.velEnd->setEnabled( ui.exportVel->isChecked() );
	ui.velIntervalsNumber->setEnabled( ui.exportVel->isChecked() );
	ShowConvValue( ui.velStart, m_dVelStart, EUnitType::VELOCITY );
	ShowConvValue( ui.velEnd, m_dVelEnd, EUnitType::VELOCITY );
	ui.velIntervalsNumber->setText( QString::number( m_nVelIntervalsNumber ) );

	// set parameters for time
	ui.timeStart->setText( QString::number( m_dTimeStart ) );
	ui.timeEnd->setText( QString::number( m_dTimeEnd ) );
	ui.timeIntervalsNumber->setText( QString::number( m_nTimeIntervalsNumber ) );

	m_bAvoidSignal = false;
}


void CSampleAnalyzerTab::ExportDataWasChanged()
{
	if ( m_bAvoidSignal ) return;

	m_dTimeStart = ui.timeStart->text().toDouble();
	m_dTimeEnd = ui.timeEnd->text().toDouble();
	m_nTimeIntervalsNumber = ui.timeIntervalsNumber->text().toInt();
	m_dPSDStart = GetConvValue( ui.psdStart, EUnitType::PARTICLE_DIAMETER );
	m_dPSDEnd = GetConvValue( ui.psdEnd, EUnitType::PARTICLE_DIAMETER );
	m_nPSDIntervalsNumber = ui.psdIntervalsNumber->text().toInt();
	m_dVelStart = GetConvValue( ui.velStart, EUnitType::VELOCITY );
	m_dVelEnd = GetConvValue( ui.velEnd, EUnitType::VELOCITY );
	m_nVelIntervalsNumber = ui.velIntervalsNumber->text().toInt();

	UpdateExportView();
}


void CSampleAnalyzerTab::VolumeDataWasChanged()
{
	if ( m_bAvoidSignal ) return;
	m_vCenter.x = GetConvValue( ui.xCoord, EUnitType::LENGTH );
	m_vCenter.y = GetConvValue( ui.yCoord, EUnitType::LENGTH );
	m_vCenter.z = GetConvValue( ui.zCoord, EUnitType::LENGTH );
	m_dRadius = GetConvValue( ui.radius, EUnitType::LENGTH );
	emit UpdateOpenGLView();
	UpdateVolumeView();
}


void CSampleAnalyzerTab::UpdateWholeView()
{
	UpdateVolumeView();
	UpdateExportView();
}

void CSampleAnalyzerTab::GetAllParticlesInRegion( double _dTime, std::vector<CSphere*>* _pAllParticles, double* _pTotalMass, double* _pTotalVolume,
	bool bPartialyInVolume, bool bConsiderBonds )
{
	*_pTotalMass = 0;
	*_pTotalVolume = 0;
	_pAllParticles->clear();
	for ( unsigned i=0; i < m_pSystemStructure->GetTotalObjectsCount(); i++ )
	{
		CPhysicalObject* pTemp = m_pSystemStructure->GetObjectByIndex( i );
		if ( pTemp != NULL )
			if ( pTemp->IsActive( _dTime ) )
			{
				if ( pTemp->GetObjectType() == SPHERE )
				{
					double dDistance = Length(pTemp->GetCoordinates(_dTime), m_vCenter);
					double dRParticle = ((CSphere*)pTemp)->GetRadius();

					if ( dDistance >= m_dRadius + dRParticle ) // particle and volume not intersect each other
						continue;
					else if ( dDistance+dRParticle <= m_dRadius ) // particle is totally in the volume
					{
						_pAllParticles->push_back( (CSphere*)pTemp );
						*_pTotalMass = ((CSphere*)pTemp)->GetMass();
						*_pTotalVolume += 4.0/3.0*PI*pow( dRParticle, 3 );
					}
					else  // particle is partially in the volume
					{
						if ( bPartialyInVolume )
							_pAllParticles->push_back( (CSphere*)pTemp );
						double dVolume = SpheresIntersectionVolume( m_dRadius, dRParticle, dDistance );
						*_pTotalVolume += dVolume;
						if ( dVolume >= 0 )
							*_pTotalMass = ((CSphere*)pTemp)->GetMass()*((CSphere*)pTemp)->GetVolume()/dVolume;
					}
				}
				else
					if ( (pTemp->GetObjectType() == SOLID_BOND)&&(bConsiderBonds == true) )
					{
						CSphere* pLeft = (CSphere*)m_pSystemStructure->GetObjectByIndex( ((CSolidBond*)pTemp)->m_nLeftObjectID );
						CSphere* pRight = (CSphere*)m_pSystemStructure->GetObjectByIndex( ((CSolidBond*)pTemp)->m_nRightObjectID );
						if ( (pLeft == NULL) || (pRight == NULL) ) continue;
						if ( (pLeft->IsActive( _dTime ) == false) || (pRight->IsActive( _dTime ) == false)) continue;
						if (Length(pLeft->GetCoordinates(_dTime), m_vCenter) + pLeft->GetRadius() <= m_dRadius)
							if (Length(pRight->GetCoordinates(_dTime), m_vCenter) + pRight->GetRadius() <= m_dRadius)
								*_pTotalVolume += m_pSystemStructure->GetBondVolume( _dTime, i );
					}
			}
	}
}



double CSampleAnalyzerTab::GetCoordinationNumber( double _dTime, std::vector<CSphere*>* _pParticles )
{
	return 0;
	/*std::vector<CSphere*> vPossibleNeighbors;
	std::vector<CPhysicalObject*> vAllSpheres = m_pSystemStructure->GetAllActiveObjects( _dTime, SPHERE );
	for (unsigned j = 0; j < vAllSpheres.size(); ++j)
	{
		double dDistance = Length(vAllSpheres[j]->GetCoordinates(_dTime), m_vCenter);
		if (dDistance >= m_dRadius + ((CSphere*)vAllSpheres[j])->GetRadius()) // particle and volume not intersect each other
			vPossibleNeighbors.push_back(((CSphere*)vAllSpheres[j]) );
	}

	CContactCalculator tempCalculator;
	for (unsigned i = 0; i < _pParticles->size(); i++)
		tempCalculator.AddParticle(i, _pParticles->at(i)->GetCoordinates(_dTime), _pParticles->at(i)->GetRadius());
	std::vector<unsigned> vID1, vID2;
	std::vector<double> vOverlaps;
	tempCalculator.GetAllOverlaps(vID1, vID2, vOverlaps);
	double dNumberOfContacts = vOverlaps.size();

	for (unsigned i = 0; i < vPossibleNeighbors.size(); i++)
		dNumberOfContacts += tempCalculator.GetContacts(vPossibleNeighbors[i]->GetCoordinates(_dTime), vPossibleNeighbors[i]->GetRadius()).size();

	// add bonds
	for ( unsigned j=0; j < m_pSystemStructure->GetTotalObjectsCount(); j++ )
	{
		CPhysicalObject* pTemp = m_pSystemStructure->GetObjectByIndex( j );
		if ( pTemp == NULL ) continue;
		if ( pTemp->IsActive( _dTime ) == false ) continue;
		if ( pTemp->GetObjectType() == SOLID_BOND )
		{
			CSolidBond* pBond = (CSolidBond*)pTemp;
			bool bConsiderBond = false;
			for ( unsigned i=0; i<_pParticles->size(); i++ )
				if (( _pParticles->at(i)->m_lObjectID == pBond->m_nLeftObjectID ) || ( _pParticles->at(i)->m_lObjectID == pBond->m_nRightObjectID ))
				{
					bConsiderBond = true;
					break;
				}
			if ( bConsiderBond )
				dNumberOfContacts++;
		}
	}


	if ( _pParticles->empty() )
		return 0;
	else
		return dNumberOfContacts/_pParticles->size();*/
}


void CSampleAnalyzerTab::UpdateMainProperties()
{
	ShowConvLabel( ui.massLabel, "Mass", EUnitType::MASS );
	SetCurrentTime(ui.currentTime->text().toDouble());

	std::vector<CSphere*> vAllParticles;
	double dTotalMass;
	double dTotalVolume;

	GetAllParticlesInRegion( m_dCurrentTime, &vAllParticles, &dTotalMass, &dTotalVolume, true, ui.considerBonds->isChecked() );
	ui.particlesInRegion->setText( QString::number( vAllParticles.size() ) );
	ShowConvValue( ui.massInRegion, dTotalMass, EUnitType::MASS );
	ui.coordinationNumber->setText( QString::number( GetCoordinationNumber( m_dCurrentTime, &vAllParticles ) ) );
	if ( m_dRadius > 0 )
		ui.porosityInRegion->setText( QString::number( 1-dTotalVolume/(4.0/3.0*PI*pow(m_dRadius, 3 )) ) );
}


bool CSampleAnalyzerTab::CheckDataCorrectness()
{
	if ( m_dRadius <= 0 )
	{
		ui.statusMessage->setText( "Radius of volume should be > 0" );
		return false;
	}

	if ( m_nTimeIntervalsNumber <= 0 )
	{
		ui.statusMessage->setText( "Error: Number of time interval should be > 0" );
		return false;
	}
	if (( m_dTimeEnd < 0 ) || ( m_dTimeStart < 0 ))
	{
		ui.statusMessage->setText( "Error: Time points should be larger than 0" );
		return false;
	}

	if ( m_dTimeEnd < m_dTimeStart )
	{
		ui.statusMessage->setText( "Error: Start time should be < end time" );
		return false;
	}

	// check corecctnes for PSD generation
	if ( ui.exportPSD->isChecked() )
	{
		if (( m_dPSDStart < 0 ) || ( m_dPSDEnd < 0 ))
		{
			ui.statusMessage->setText( "Error: PSD generation diameters should be > 0" );
			return false;
		}
		if ( m_dPSDStart > m_dPSDEnd )
		{
			ui.statusMessage->setText( "Error: PSD end diameter should be > start diameter" );
			return false;
		}
		if ( m_nPSDIntervalsNumber <= 0 )
		{
			ui.statusMessage->setText( "Error: Number of PSD intervals should be > 0" );
			return false;
		}
	}

	// check corecctnes for Vel generation
	if ( ui.exportVel->isChecked() )
	{
		if (( m_dVelStart < 0 ) || ( m_dVelEnd < 0 ) )
		{
			ui.statusMessage->setText( "Error: Export velocities should be > 0" );
			return false;
		}
		if ( m_dVelStart > m_dVelEnd )
		{
			ui.statusMessage->setText( "Error: End velocity should be > start velocity" );
			return false;
		}
		if ( m_nVelIntervalsNumber <= 0 )
		{
			ui.statusMessage->setText( "Error: Number of velocity intervals should be > 0" );
			return false;
		}
	}

	ui.statusMessage->setText("");
	return true;
}


void CSampleAnalyzerTab::ExportData()
{
	if ( CheckDataCorrectness() == false ) return;

	QString sFileName = QFileDialog::getSaveFileName(this, tr("Export results"), "", tr( "Results data (*.csv);;All files (*.*);;" ));
	if ( sFileName.simplified() == "" )
		return;

	std::ofstream outFile;
	outFile.open(UnicodePath(qs2ss(sFileName))); //open a file

	ui.exportData->setText( "Exporting..." );
	ui.statusMessage->setText("Exporting started");
	// separator
	std::string sSep = "; ";

	// print header
	outFile<<"Time [s]" << sSep;
	if ( ui.exportMass->isChecked() )
		outFile << "Mass [kg]" << sSep;
	if ( ui.exportPorosity->isChecked() )
		outFile << "Porosity [-]" << sSep;
	if ( ui.exportParticlesNumber->isChecked() )
		outFile << "Number [-]" << sSep;
	if ( ui.exportCoordNumber->isChecked() )
		outFile << "Coord_Number [-] " << sSep;

	if (( ui.exportPSD->isChecked() ) && ( m_nPSDIntervalsNumber != 0))
		for ( unsigned i=0; i < m_nPSDIntervalsNumber; i++ )
			outFile << "d=" << m_dPSDStart + (i+0.5)*(m_dPSDEnd-m_dPSDStart)/m_nPSDIntervalsNumber<< " [m]" << sSep;

	if (( ui.exportVel->isChecked() ) && ( m_nVelIntervalsNumber != 0))
		for ( unsigned i=0; i < m_nVelIntervalsNumber; i++ )
			outFile << "V=" << m_dVelStart + (i+0.5)*(m_dVelEnd-m_dVelStart)/m_nVelIntervalsNumber<< " [m/s]" << sSep;
	outFile << std::endl;

	// export data
	double dCurrentTime;
	double dTotalMass, dTotalVolume, dPorosity;
	double dVolumeRegion = 4.0/3.0*PI*pow(m_dRadius, 3 );
	double dIntervalSize = (m_dPSDEnd - m_dPSDStart)/m_nPSDIntervalsNumber;
	double dIntervalVel = (m_dVelEnd - m_dVelStart)/m_nVelIntervalsNumber;

	std::vector<CSphere*> vAllParticles;
	for ( unsigned i=0; i < m_nTimeIntervalsNumber+1; i++ )
	{
		if ( m_nVelIntervalsNumber != 0 )
			dCurrentTime = m_dTimeStart+i*(m_dTimeEnd - m_dTimeStart)/m_nTimeIntervalsNumber;
		else
			dCurrentTime = m_dTimeStart;

		GetAllParticlesInRegion( dCurrentTime, &vAllParticles, &dTotalMass, &dTotalVolume );
		if ( dVolumeRegion > 0 )
			dPorosity = 1-dTotalVolume/dVolumeRegion;
		else
			dPorosity = 0;

		outFile<< dCurrentTime << sSep;
		if ( ui.exportMass->isChecked() )
			outFile << dTotalMass << sSep;
		if ( ui.exportPorosity->isChecked() )
			outFile << dPorosity << sSep;
		if ( ui.exportParticlesNumber->isChecked() )
			outFile << vAllParticles.size() << sSep;
		if (  ui.exportCoordNumber->isChecked() )
			outFile << GetCoordinationNumber( dCurrentTime, &vAllParticles );

		std::vector<double> velDistribution( m_nVelIntervalsNumber, 0 );
		std::vector<double> sizeDistribution( m_nPSDIntervalsNumber, 0 );
		for ( unsigned j=0; j < vAllParticles.size(); j++ )
		{
			double dDiameter = vAllParticles[ j ]->GetRadius()*2;
			double dVelocity = vAllParticles[j]->GetVelocity(dCurrentTime).Length();
			int nOffsetDiameter = (dDiameter - m_dPSDStart)/dIntervalSize;
			int nOffsetVel = (dVelocity - m_dVelStart)/dIntervalVel;
			if ( nOffsetVel < 0 ) nOffsetVel = 0;
			if ( nOffsetVel >= velDistribution.size() ) nOffsetVel = (int)velDistribution.size()-1;
			if ( nOffsetDiameter < 0 ) nOffsetDiameter = 0;
			if ( nOffsetDiameter >= sizeDistribution.size() ) nOffsetDiameter = (int)sizeDistribution.size()-1;
			velDistribution[ nOffsetVel ] ++;
			sizeDistribution[ nOffsetDiameter ] ++;
		}
		if ( ui.exportPSD->isChecked() )
			for ( unsigned j=0; j < sizeDistribution.size(); j++ )
				outFile << sizeDistribution[ j ] << sSep;

		if ( ui.exportVel->isChecked() )
			for ( unsigned j=0; j < velDistribution.size(); j++ )
				outFile << velDistribution[ j ] << sSep;

		outFile << std::endl;
	}
	outFile.close();

	ui.statusMessage->setText("Data has been sucsesfully generated");
	ui.exportData->setText( "Exporting data" );
}