/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_GeometriesDatabaseTab.h"

class CGeometriesDatabaseTab : public CMusenDialog
{
	Q_OBJECT

	Ui::geometriesDatabaseTab ui{};
	QString m_lastUsedFilePath;		// Last used full file path to use in file dialogs.
	bool m_isDBModified{};			// Whether the database was modified after saving/loading.

public:
	CGeometriesDatabaseTab(QWidget *parent = nullptr);

private:
	void InitializeConnections();
	void Initialize() override;
	void SetupScaleButton();

	void UpdateWholeView() override;
	void UpdateWindowTitle();
	void UpdateGeometriesList() const;
	void UpdateGeometryInfoHeaders() const;
	void UpdateGeometryInfo() const;
	void Update3DView() const;
	void UpdateButtons() const;

	QString DefaultPath() const;
	void SetDBModified(bool _modified);

	void keyPressEvent(QKeyEvent* _event) override;
	void closeEvent(QCloseEvent* _event) override;

private slots:
	void NewDatabase();
	void LoadDatabase();
	void SaveDatabase();
	void SaveDatabaseAs();

	void ImportGeometry();
	void ExportGeometry();
	void DeleteGeometry();
	void UpGeometry();
	void DownGeometry();
	void ScaleGeometry(double _factor);

	void GeometrySelected() const;
	void GeometryRenamed();

signals:
	void GeometryAdded();
};
