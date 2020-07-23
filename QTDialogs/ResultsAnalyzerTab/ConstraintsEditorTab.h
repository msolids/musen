/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_ConstraintsEditorTab.h"
#include "Constraints.h"

class CConstraintsEditorTab : public CMusenDialog
{
	Q_OBJECT

private:
	CConstraints *m_pConstraints;

public:
	CConstraintsEditorTab(QWidget *parent = 0);
	~CConstraintsEditorTab();

	void UpdateSettings();

	void SetConstraintsPtr(CConstraints *_pConstraints);

	void SetMaterialsVisible(bool _bVisible);
	void SetMaterials2Visible(bool _bVisible);
	void SetVolumesVisible(bool _bVisible);
	void SetGeometriesVisible(bool _bVisible);
	void SetDiametersVisible(bool _bVisible);
	void SetDiameters2Visible(bool _bVisible);

	void SetWidgetsEnabled(bool _bEnabled);

private:
	Ui::CConstraintsEditorTab ui;

	void InitializeConnections();

	void UpdateMaterials();
	void UpdateVolumes();
	void UpdateGeometries();
	void UpdateDiameters();
	void UpdateIterativeShifts();

public slots:
	void UpdateWholeView();

private slots:
	void NewVolumeChecked(QListWidgetItem* _pItem);
	void NewGeometryChecked(QListWidgetItem* _pItem);
	void NewMaterialChecked(QListWidgetItem* _pItem);
	void NewMaterialSelected(int _nRow);
	void NewMaterial2Checked(QListWidgetItem* _pItem);
	void NewDiameterEntered(int _nRow, int _nCol);
	void MaterialsActivated(bool _bActive);
	void VolumesActivated(bool _bActive);
	void GeometriesActivated(bool _bActive);
	void DiametersActivated(bool _bActive);
};
