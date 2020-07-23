/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SelectiveSavingTab.h"

CSelectiveSavingTab::CSelectiveSavingTab(CSimulatorManager* _pSimManager, QWidget* parent): CMusenDialog(parent)
{
	ui.setupUi(this);
	m_pSimulatorManager = _pSimManager;
	InitializeConnections();
}

void CSelectiveSavingTab::InitializeConnections() const
{
	// signals about models
	connect(ui.pushButtonOk, &QPushButton::clicked, this, &CSelectiveSavingTab::accept);
	// particles
	connect(ui.groupBoxPropertiesP,   &QGroupBox::toggled,		this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxCoordinates,	  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxVelocity,	  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxAngVelocity,	  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxForce,		  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxQuaternion,	  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxTensor,		  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	// solid bonds
	connect(ui.groupBoxPropertiesSB,  &QGroupBox::toggled,		this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxSBForce,		  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxSBTangOverlap, &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxSBTotTorque,	  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	// liquid bonds
	connect(ui.groupBoxPropertiesLB,  &QGroupBox::toggled,		this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxLBForce,		  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	// triangular walls
	connect(ui.groupBoxPropertiesTW,  &QGroupBox::toggled,		this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxTWCoordinates, &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxTWForce,		  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
	connect(ui.checkBoxTWVelocity,	  &QCheckBox::stateChanged, this, &CSelectiveSavingTab::SetParameters);
}

void CSelectiveSavingTab::UpdateWholeView()
{
	const SSelectiveSavingFlags tmp = m_pSimulatorManager->GetSimulatorPtr()->GetSelectiveSavingFlags();
	// particles
	ui.checkBoxCoordinates->setChecked(tmp.bCoordinates);
	ui.checkBoxVelocity->setChecked(tmp.bVelocity);
	ui.checkBoxAngVelocity->setChecked(tmp.bAngVelocity);
	ui.checkBoxForce->setChecked(tmp.bForce);
	ui.checkBoxQuaternion->setChecked(tmp.bQuaternion);
	ui.checkBoxTensor->setChecked(tmp.bTensor);
	// solid bonds
	ui.checkBoxSBForce->setChecked(tmp.bSBForce);
	ui.checkBoxSBTangOverlap->setChecked(tmp.bSBTangOverlap);
	ui.checkBoxSBTotTorque->setChecked(tmp.bSBTotTorque);
	// liquid bonds
	ui.checkBoxLBForce->setChecked(tmp.bLBForce);
	// triangular walls
	ui.checkBoxTWCoordinates->setChecked(tmp.bTWPlaneCoord);
	ui.checkBoxTWForce->setChecked(tmp.bTWForce);
	ui.checkBoxTWVelocity->setChecked(tmp.bTWVelocity);
}

void CSelectiveSavingTab::SetParameters()
{
	// particles
	if (ui.groupBoxPropertiesP->isChecked())
	{
		m_SSelectiveSavingFlags.bCoordinates   = ui.checkBoxCoordinates->isChecked();
		m_SSelectiveSavingFlags.bAngVelocity   = ui.checkBoxAngVelocity->isChecked();
		m_SSelectiveSavingFlags.bForce         = ui.checkBoxForce->isChecked();
		m_SSelectiveSavingFlags.bQuaternion    = ui.checkBoxQuaternion->isChecked();
		m_SSelectiveSavingFlags.bVelocity      = ui.checkBoxVelocity->isChecked();
		m_SSelectiveSavingFlags.bTensor        = ui.checkBoxTensor->isChecked();
	}
	else
		m_SSelectiveSavingFlags.SetAllParticles(false);

	// solid bonds
	if (ui.groupBoxPropertiesSB->isChecked())
	{
		m_SSelectiveSavingFlags.bSBForce       = ui.checkBoxSBForce->isChecked();
		m_SSelectiveSavingFlags.bSBTangOverlap = ui.checkBoxSBTangOverlap->isChecked();
		m_SSelectiveSavingFlags.bSBTotTorque   = ui.checkBoxSBTotTorque->isChecked();
	}
	else
		m_SSelectiveSavingFlags.SetAllSolidBonds(false);

	// liquid bonds
	if (ui.groupBoxPropertiesLB->isChecked())
		m_SSelectiveSavingFlags.bLBForce       = ui.checkBoxLBForce->isChecked();
	else
		m_SSelectiveSavingFlags.SetAllLiquidBonds(false);

	// triangular walls
	if (ui.groupBoxPropertiesTW->isChecked())
	{
		m_SSelectiveSavingFlags.bTWPlaneCoord  = ui.checkBoxTWCoordinates->isChecked();
		m_SSelectiveSavingFlags.bTWForce       = ui.checkBoxTWForce->isChecked();
		m_SSelectiveSavingFlags.bTWVelocity    = ui.checkBoxTWVelocity->isChecked();
	}
	else
		m_SSelectiveSavingFlags.SetAllWalls(false);

	m_pSimulatorManager->GetSimulatorPtr()->SetSelectiveSaving(true);
	m_pSimulatorManager->GetSimulatorPtr()->SetSelectiveSavingParameters(m_SSelectiveSavingFlags);

	UpdateWholeView();
}