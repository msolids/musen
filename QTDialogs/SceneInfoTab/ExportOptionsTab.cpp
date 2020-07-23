/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ExportOptionsTab.h"

CExportOptionsTab::CExportOptionsTab( QWidget *parent )
	:QDialog( parent )
{
	ui.setupUi(this);

	m_dStartTime = 0;
	m_dEndTime = 1e-3;
	m_dTimeStep = 1e-4;
	m_nNumberOfPSDClasses = 20;
	m_sSeparator = ";";
}

void CExportOptionsTab::UpdateEntriesData()
{
	ui.startTime->setText( QString::number( m_dStartTime ) );
	ui.endTime->setText( QString::number( m_dEndTime ) );
	ui.timeStep->setText( QString::number( m_dTimeStep ) );
	ui.separator->setText( m_sSeparator );
	ui.numberOfPSDClasses->setText( QString::number( m_nNumberOfPSDClasses ) );
}


void CExportOptionsTab::UpdateWholeView()
{
	UpdateEntriesData();
}

void CExportOptionsTab::setVisible( bool _bVisible )
{
	UpdateWholeView();
	QDialog::setVisible( _bVisible );
}


void CExportOptionsTab::InitializeConnections()
{
	QObject::connect( ui.startTime, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
	QObject::connect( ui.endTime, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
	QObject::connect( ui.timeStep, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
	QObject::connect( ui.separator, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
	QObject::connect( ui.numberOfPSDClasses, SIGNAL(editingFinished()), this, SLOT(DataWasChanged()) );
}


void CExportOptionsTab::DataWasChanged()
{
	m_dStartTime = ui.startTime->text().toDouble();
	if ( m_dStartTime < 0 ) m_dStartTime = 0;

	m_dEndTime  = ui.endTime->text().toDouble();
	if ( m_dEndTime < 0 ) m_dEndTime = 0;

	m_dTimeStep = ui.timeStep->text().toDouble();
	if ( m_dTimeStep < 0 ) m_dTimeStep = 0;


	m_nNumberOfPSDClasses = ui.numberOfPSDClasses->text().toInt();
	if ( m_nNumberOfPSDClasses <= 0 ) m_nNumberOfPSDClasses = 1;

	m_sSeparator = ui.separator->text();
	UpdateEntriesData();
}