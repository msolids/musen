/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SceneEditorTab.h"
#include <QMessageBox>

CSceneEditorTab::CSceneEditorTab(QWidget *parent /*= 0*/) : CMusenDialog(parent)
{
	ui.setupUi(this);

	QVector<QWidget*> labels;
	labels.push_back(ui.setCenterOfMass);
	labels.push_back(ui.labelRotation);
	labels.push_back(ui.labelAngle);
	labels.push_back(ui.labelOffset);
	labels.push_back(ui.labelVelocity);
	int maxWidth = (*std::max_element(labels.begin(), labels.end(), [](const QWidget* l, const QWidget* r) { return (l->width() < r->width()); }))->width();
	for (QWidget* w : labels)
		w->setFixedWidth(maxWidth);

	m_bAllowUndoFunction = false;

	connect(ui.rotateSystem,	&QPushButton::clicked, this, &CSceneEditorTab::RotateSystem);
	connect(ui.undoRotation,	&QPushButton::clicked, this, &CSceneEditorTab::UndoRotation);
	connect(ui.setCenterOfMass, &QPushButton::clicked, this, &CSceneEditorTab::SetCenterOfMass);
	connect(ui.moveSystem,		&QPushButton::clicked, this, &CSceneEditorTab::MoveSystem);
	connect(ui.setVelocity,		&QPushButton::clicked, this, &CSceneEditorTab::SetVelocity);

	connect(ui.groupBoxPBC,		&QGroupBox::toggled, this, &CSceneEditorTab::SetPBC);

	connect(ui.checkBoxPBCX, &QCheckBox::stateChanged, this, &CSceneEditorTab::SetPBC);
	connect(ui.checkBoxPBCY, &QCheckBox::stateChanged, this, &CSceneEditorTab::SetPBC);
	connect(ui.checkBoxPBCZ, &QCheckBox::stateChanged, this, &CSceneEditorTab::SetPBC);

	connect(ui.lineEditPBCMinX, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditPBCMinY, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditPBCMinZ, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditPBCMaxX, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditPBCMaxY, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditPBCMaxZ, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditVelocityX, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditVelocityY, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);
	connect(ui.lineEditVelocityZ, &QLineEdit::editingFinished, this, &CSceneEditorTab::SetPBC);

	connect(ui.checkBoxAnisotropy, &QCheckBox::stateChanged, this, &CSceneEditorTab::SetAnisotropy);
	connect(ui.checkBoxContactRadius, &QCheckBox::stateChanged, this, &CSceneEditorTab::SetContactRadius);
}

void CSceneEditorTab::setVisible( bool _bVisible )
{
	if (!_bVisible)
		m_bAllowUndoFunction = false;
	CMusenDialog::setVisible(_bVisible);
}

void CSceneEditorTab::UpdateWholeView()
{
	ui.undoRotation->setEnabled(m_bAllowUndoFunction);
	ShowConvLabel(ui.labelCenterX, "X", EUnitType::LENGTH);
	ShowConvLabel(ui.labelCenterY, "Y", EUnitType::LENGTH);
	ShowConvLabel(ui.labelCenterZ, "Z", EUnitType::LENGTH);
	ShowConvLabel(ui.labelOffsetX, "X", EUnitType::LENGTH);
	ShowConvLabel(ui.labelOffsetY, "Y", EUnitType::LENGTH);
	ShowConvLabel(ui.labelOffsetZ, "Z", EUnitType::LENGTH);
	ShowConvLabel(ui.labelVelocityX, "Vx", EUnitType::VELOCITY);
	ShowConvLabel(ui.labelVelocityY, "Vy", EUnitType::VELOCITY);
	ShowConvLabel(ui.labelVelocityZ, "Vz", EUnitType::VELOCITY);
	ShowConvLabel(ui.labelPBCMin, "Min", EUnitType::LENGTH);
	ShowConvLabel(ui.labelPBCMax, "Max", EUnitType::LENGTH);
	ShowConvLabel(ui.labelPBCVel, "Velocity", EUnitType::VELOCITY);

	UpdatePBC();

	ui.checkBoxAnisotropy->setChecked(m_pSystemStructure->IsAnisotropyEnabled());
	ui.checkBoxContactRadius->setChecked(m_pSystemStructure->IsContactRadiusEnabled());
}

void CSceneEditorTab::SetCenterOfMass()
{
	ShowConvValue(ui.lineEditCenterX, ui.lineEditCenterY, ui.lineEditCenterZ, m_pSystemStructure->GetCenterOfMass(0), EUnitType::LENGTH);
}

void CSceneEditorTab::RotateSystem()
{
	m_RotationCenter = GetConvValue(ui.lineEditCenterX, ui.lineEditCenterY, ui.lineEditCenterZ, EUnitType::LENGTH);
	m_RotationAngle = GetConvValue(ui.lineEditAngleX, ui.lineEditAngleY, ui.lineEditAngleZ, EUnitType::NONE ) * PI / 180;

	m_pSystemStructure->ClearAllStatesFrom(0);
	m_pSystemStructure->RotateSystem(0, m_RotationCenter, m_RotationAngle);
	m_bAllowUndoFunction = true;
	UpdateWholeView();
	emit UpdateOpenGLView();
}

void CSceneEditorTab::MoveSystem()
{
	m_pSystemStructure->ClearAllStatesFrom(0);
	m_pSystemStructure->MoveSystem(0, GetConvValue(ui.lineEditOffsetX, ui.lineEditOffsetY, ui.lineEditOffsetZ, EUnitType::LENGTH));
	UpdateWholeView();
	emit UpdateOpenGLView();
}

void CSceneEditorTab::SetVelocity()
{
	m_pSystemStructure->ClearAllStatesFrom(0);
	m_pSystemStructure->SetSystemVelocity(0, GetConvValue(ui.lineEditVelocityX, ui.lineEditVelocityY, ui.lineEditVelocityZ, EUnitType::VELOCITY));
	UpdateWholeView();
	emit UpdateOpenGLView();
}

void CSceneEditorTab::UndoRotation()
{
	m_pSystemStructure->RotateSystem(0, m_RotationCenter, m_RotationAngle * (-1));
	UpdateWholeView();
	emit UpdateOpenGLView();
}


void CSceneEditorTab::SetPBC()
{
	if (m_bAvoidSignal) return;
	if (m_pSystemStructure->GetNumberOfSpecificObjects(SOLID_BOND) != 0)
		QMessageBox::information(this, ("Bonds over PBC"), ("Modification of PBC will influence existing solid bonds!"), QMessageBox::Ok);
	SPBC pbcNew;
	pbcNew.bEnabled = ui.groupBoxPBC->isChecked();
	pbcNew.bX = ui.checkBoxPBCX->isChecked();
	pbcNew.bY = ui.checkBoxPBCY->isChecked();
	pbcNew.bZ = ui.checkBoxPBCZ->isChecked();

	pbcNew.SetDomain(GetConvValue(ui.lineEditPBCMinX, ui.lineEditPBCMinY, ui.lineEditPBCMinZ, EUnitType::LENGTH), GetConvValue(ui.lineEditPBCMaxX, ui.lineEditPBCMaxY, ui.lineEditPBCMaxZ, EUnitType::LENGTH));
	pbcNew.vVel = GetConvValue(ui.lineEditVelX, ui.lineEditVelY, ui.lineEditVelZ, EUnitType::VELOCITY);
	m_pSystemStructure->SetPBC(pbcNew);

	UpdateWholeView();
	emit UpdateOpenGLView();
}

void CSceneEditorTab::UpdatePBC()
{
	m_bAvoidSignal = true;

	const SPBC& pbc = m_pSystemStructure->GetPBC();
	ui.groupBoxPBC->setChecked(pbc.bEnabled);

	ShowConvValue( ui.lineEditPBCMinX, ui.lineEditPBCMinY, ui.lineEditPBCMinZ, pbc.initDomain.coordBeg, EUnitType::LENGTH);
	ShowConvValue( ui.lineEditPBCMaxX, ui.lineEditPBCMaxY, ui.lineEditPBCMaxZ, pbc.initDomain.coordEnd, EUnitType::LENGTH);
	ShowConvValue( ui.lineEditVelX, ui.lineEditVelY, ui.lineEditVelZ, pbc.vVel, EUnitType::VELOCITY);

	ui.checkBoxPBCX->setChecked(pbc.bX);
	ui.lineEditPBCMinX->setEnabled(pbc.bX && pbc.bEnabled);
	ui.lineEditPBCMaxX->setEnabled(pbc.bX && pbc.bEnabled);
	ui.lineEditVelX->setEnabled(pbc.bX && pbc.bEnabled);

	ui.checkBoxPBCY->setChecked(pbc.bY);
	ui.lineEditPBCMinY->setEnabled(pbc.bY && pbc.bEnabled);
	ui.lineEditPBCMaxY->setEnabled(pbc.bY && pbc.bEnabled);
	ui.lineEditVelY->setEnabled(pbc.bY && pbc.bEnabled);

	ui.checkBoxPBCZ->setChecked(pbc.bZ);
	ui.lineEditPBCMinZ->setEnabled(pbc.bZ && pbc.bEnabled);
	ui.lineEditPBCMaxZ->setEnabled(pbc.bZ && pbc.bEnabled);
	ui.lineEditVelZ->setEnabled(pbc.bZ && pbc.bEnabled);

	m_bAvoidSignal = false;
}

void CSceneEditorTab::SetAnisotropy()
{
	m_pSystemStructure->EnableAnisotropy(ui.checkBoxAnisotropy->isChecked());
}

void CSceneEditorTab::SetContactRadius()
{
	m_pSystemStructure->EnableContactRadius(ui.checkBoxContactRadius->isChecked());
	emit ContactRadiusEnabled();
}
