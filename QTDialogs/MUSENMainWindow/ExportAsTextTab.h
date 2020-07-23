/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_ExportAsTextTab.h"
#include "ExportAsText.h"
#include "ConstraintsEditorTab.h"
#include <QTimer>

class CExportAsTextThread : public QObject
{
	Q_OBJECT

	CExportAsText* m_pExporter;

public:
	CExportAsTextThread(CExportAsText* _exporter, QObject* parent = nullptr);

public slots:
	void StartExporting();
	void StopExporting() const;

signals:
	void finished();
};

class CExportAsTextTab : public CMusenDialog
{
	Q_OBJECT

	Ui::CExportAsTextTab ui;

	std::vector<double> m_vTimePoints;  // List of time points which should be considered.

	CConstraints		   m_constraints;
	CExportAsText          m_exporter;
	QThread*               m_pQTThread;
	CExportAsTextThread*   m_pExportThread;
	QTimer				   m_UpdateTimer;

public:
	CExportAsTextTab(QWidget *parent = Q_NULLPTR);

	void SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB) override;

	void UpdateWholeView() override;

private:
	// Connects qt objects to slots.
	void InitializeConnections() const;
	// Update the value of precision.
	void UpdatePrecision() const;
	// Creates vector of time points, depending on time parameters and saving mode.
	void CalculateTimePoints();
	// Updates size of time points.
	void UpdateTimeParameters() const;
	// Sets all flags, which related to all checkboxes.
	void ApplyAllFlags();

private slots:
	void SetObjectTypeCheckBoxes(bool _active) const;
	void SelectiveSavingToggled();
	void SetNewTime();
	void UpdateTimeFromSimulation();
	void UpdateProgressInfo() const;
	void ExportPressed();
	void ExportingFinished();
	void SetRelevantCheckBoxesParticle() const;
	void SetRelevantCheckBoxesSB() const;
	void SetRelevantCheckBoxesTW() const;
	void SetQuaternionCheckBox();
	void SetWholeTabEnabled(bool _enabled) const;

signals:
	void RunningStatusChanged(ERunningStatus _status);
};