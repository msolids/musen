/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_GeometriesDatabaseTab.h"
#include "GeomOpenGLView.h"
#include "GeometriesDatabase.h"

class CGeometriesDatabaseTab : public CMusenDialog
{
	Q_OBJECT

private:
	Ui::geometriesDatabaseTab ui;

public:
	CGeometriesDatabaseTab(QWidget *parent = 0);

private:
	void InitializeConnections();
	void UpdateButtons();
	void UpdateSelectedGeomInfo();

private slots:
	void NewDatabase();
	void SaveDatabase();
	void SaveDatabaseAs( const QString& _sFileName = "" );
	void LoadDatabase();
	void DataWasChanged();
	void ShowGeometry();
	void DeleteGeometry();

	void AddNewGeometry(); 
	void ExportGeometry();
	void NewGeometryAdded();
	void UpdateWholeView();
	void NewRowSelected();
	void UpGeometry();
	void DownGeometry();

signals:
	void GeometryAdded();
};
