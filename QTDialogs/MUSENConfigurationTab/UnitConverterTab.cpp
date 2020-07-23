/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "UnitConverterTab.h"
#include "qtOperations.h"

CUnitConvertorTab::CUnitConvertorTab( QSettings* _pSettings, QWidget *parent )
	:CMusenDialog( parent ), m_pSettings( _pSettings )
{
	ui.setupUi( this );
	m_bAvoidSignal = false;

	LoadConfiguration();
	InitializeConnections();

}

void CUnitConvertorTab::InitializeConnections()
{
	connect(ui.applyUnits, SIGNAL(clicked()), this, SLOT(SetNewUnits()));
	connect(ui.pushButtonOk, SIGNAL(clicked()), this, SLOT(SetNewUnitsAndClose()));
	connect(ui.restoreDefault, SIGNAL(clicked()), this, SLOT(RestoreDefaultUnits()));
}

void CUnitConvertorTab::SetPointers( CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor,
	CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB )
{
	m_pUnitConvertor = _pUnitConvertor;
	ui.table->setRowCount(m_pUnitConvertor->GetPropertiesNumber());

	m_pUnitConvertor->SetSelectedUnitType(EUnitType::MASS, m_pSettings->value(UC_MASS_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::MASS_STREAM, m_pSettings->value(UC_MASS_STREAM_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::TEMPERATURE, m_pSettings->value(UC_TEMPERATURE_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::TIME, m_pSettings->value(UC_TIME_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::LENGTH, m_pSettings->value(UC_LENGTH_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::PARTICLE_DIAMETER, m_pSettings->value(UC_PARTICLE_DIAMETER_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::PRESSURE, m_pSettings->value(UC_PRESSURE_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::VELOCITY, m_pSettings->value(UC_VELOCITY_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::FORCE, m_pSettings->value(UC_FORCE_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::VOLUME, m_pSettings->value(UC_VOLUME_UNIT).toInt());
	m_pUnitConvertor->SetSelectedUnitType(EUnitType::ANGULAR_VELOCITY, m_pSettings->value(UC_ANGULAR_VELOCITY_UNIT).toInt());

	for (int i = 0; i < ui.table->rowCount(); i++)
	{
		ui.table->setItem(i, 0, new QTableWidgetItem(ss2qs(m_pUnitConvertor->GetPropertyNameByIndex(i))));
		ui.table->item(i, 0)->setFlags(ui.table->item(i, 0)->flags() | Qt::ItemIsEditable);

		ui.table->setItem(i, 1, new QTableWidgetItem());
		QComboBox* pCombo = new QComboBox(this);
		std::vector<std::string> vVariants = m_pUnitConvertor->GetPossibleUnitsByIndex(i);
		for (unsigned j = 0; j < vVariants.size(); j++)
			pCombo->insertItem(j, ss2qs(vVariants[j]));
		m_pComboBoxes.push_back(pCombo);

		ui.table->setCellWidget(i, 1, pCombo);
	}
	ui.table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	UpdateWholeView();
}


void CUnitConvertorTab::SaveConfiguration()
{
	m_pSettings->setValue(UC_MASS_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::MASS));
	m_pSettings->setValue(UC_MASS_STREAM_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::MASS_STREAM));
	m_pSettings->setValue(UC_TEMPERATURE_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::TEMPERATURE));
	m_pSettings->setValue(UC_TIME_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::TIME));
	m_pSettings->setValue(UC_LENGTH_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::LENGTH));
	m_pSettings->setValue(UC_PARTICLE_DIAMETER_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::PARTICLE_DIAMETER));
	m_pSettings->setValue(UC_PRESSURE_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::PRESSURE));
	m_pSettings->setValue(UC_VELOCITY_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::VELOCITY));
	m_pSettings->setValue(UC_FORCE_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::FORCE));
	m_pSettings->setValue(UC_VOLUME_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::VOLUME));
	m_pSettings->setValue(UC_ANGULAR_VELOCITY_UNIT, m_pUnitConvertor->GetSelectedUnitType(EUnitType::ANGULAR_VELOCITY));
}

void CUnitConvertorTab::LoadConfiguration()
{
}


void CUnitConvertorTab::UpdateWholeView()
{
	for (unsigned i = 0; i < m_pComboBoxes.size(); i++)
		m_pComboBoxes[i]->setCurrentIndex(m_pUnitConvertor->GetSelectedUnitTypeByIndex(i));
}

void CUnitConvertorTab::SetNewUnits()
{
	for (unsigned i = 0; i < m_pComboBoxes.size(); i++)
		m_pUnitConvertor->SetSelectedUnitTypeByIndex(i, m_pComboBoxes[i]->currentIndex());
	emit NewUnitsSelected();
}

void CUnitConvertorTab::SetNewUnitsAndClose()
{
	SetNewUnits();
	accept();
}

void CUnitConvertorTab::RestoreDefaultUnits()
{
	m_pUnitConvertor->RestoreDefaultUnits();
	UpdateWholeView();
	emit NewUnitsSelected();
}