/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_SceneEditorTab.h"
#include "GeneralMUSENDialog.h"
#include "UnitConvertor.h"

class CSceneEditorTab: public CMusenDialog
{
	Q_OBJECT

private:
	Ui::sceneEditorTab ui;
	bool m_bAllowUndoFunction; // allow undo function

	CVector3 m_RotationCenter;
	CVector3 m_RotationAngle;

public:
	CSceneEditorTab(QWidget *parent = 0);

public slots:
	void setVisible( bool _bVisible );
	void UpdateWholeView();

	// rotation
	void RotateSystem();
	void SetCenterOfMass();
	void UndoRotation();
	// movement
	void MoveSystem();
	// velocity
	void SetVelocity();
	void SetPBC();

private:
	void UpdatePBC();

private slots:
	void SetAnisotropy();
	void SetContactRadius();
	// Called when Reset Bonds button is activated.
	void ResetBonds() const;

signals:
	void ContactRadiusEnabled();
};