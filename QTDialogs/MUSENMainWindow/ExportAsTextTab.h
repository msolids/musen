/* Copyright (c) 2013-2022, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_ExportAsTextTab.h"
#include "ExportAsText.h"
#include "ConstraintsEditorTab.h"
#include <QTimer>

class CExportWorker : public QObject
{
	Q_OBJECT

	CExportAsText* m_exporter;

public:
	CExportWorker(CExportAsText* _exporter, QObject* _parent = nullptr);

public slots:
	void StartExporting();
	void StopExporting() const;

signals:
	void finished();
};

class CExportAsTextTab : public CMusenDialog
{
	Q_OBJECT

	Ui::CExportAsTextTab ui{};

	CExportAsText  m_exporter;       // Text exporter itself.
	CConstraints   m_constraints;    // Constraints for text exporter.
	QThread*       m_exportThread{}; // Tread where export worker is running.
	CExportWorker* m_exportWorker{}; // Runner to execute exporter.
	QTimer         m_updateTimer;    // Timer to update user interface.

	CPackageGenerator* m_packageGenerator{ nullptr }; // Pointer to actual package generator.
	CBondsGenerator*   m_bondsGenerator{ nullptr };   // Pointer to actual bonds generator.

public:
	CExportAsTextTab(CPackageGenerator* _pakageGenerator, CBondsGenerator* _bondsGenerator, QWidget* _parent = nullptr);

	// Sets all pointers to all required data. Must be called before any other function.
	void SetPointers(CSystemStructure* _systemStructure, CUnitConvertor* _unitConvertor, CMaterialsDatabase* _materialsDB, CGeometriesDatabase* _geometriesDB, CAgglomeratesDatabase* _agglomeratesDB) override;

	// Is called when visibility of the widget changes.
	void setVisible(bool _visible) override;

	// Updates the whole widget.
	void UpdateWholeView() override;

private:
	// Connects qt objects to slots.
	void InitializeConnections() const;

	// Is called when user switches between save all and save selective.
	void UpdateAllFlags();
	// Enable/disables orientation check box.
	void UpdateOrientationFlag() const;
	// Update the value of precision.
	void UpdatePrecision() const;
	// Updates time from user input.
	void UpdateTime();
	// Updates time from simulation parameters.
	void UpdateTimeFromSimulation();
	// Updates progress.
	void UpdateProgressInfo() const;

	// Sets activity of object-related widgets.
	void SetEnabledObjectWidgets(bool _active) const;
	// Sets activity of time-dependent object-related widgets.
	void SetEnabledTDWidgets() const;
	// Disables the whole tab during exporting.
	void SetEnabledAll(bool _enabled) const;

	// Returns vector of time points, depending on time parameters and saving mode.
	std::vector<double> CalculateTimePoints();
	// Sets all selected flags to exporter.
	void ApplyAllFlags();

	// Is called when export starts.
	void ExportPressed();
	// Is called when export finishes.
	void ExportingFinished();

signals:
	void RunningStatusChanged(ERunningStatus _status);
};