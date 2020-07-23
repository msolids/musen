/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzer.h"
#include "GeneralMUSENDialog.h"
#include "ui_ResultsAnalyzerTab.h"
#include "ConstraintsEditorTab.h"
#include <QThread>
#include <QTimer>

class CAnalyzerThread : public QObject
{
	Q_OBJECT
public:
	CResultsAnalyzer* m_pAnalyzer;
private:
	QString m_sFileName;
	QThread m_Thread;
public:
	CAnalyzerThread(CResultsAnalyzer *_pAnalyzer, QObject *parent = 0);
	~CAnalyzerThread();
	void Run(const QString& _sFileName);
	void Stop();
	void StopAnalyzing();
private slots:
	void StartAnalyzing();
signals:
	void Finished();
};

class CResultsAnalyzerTab : public CMusenDialog
{
	Q_OBJECT

protected:
	Ui::CResultsAnalyzerTab ui;
	CResultsAnalyzer *m_pAnalyzer;
	CConstraintsEditorTab *m_pConstraintsEditorTab;

private:
	std::vector<CResultsAnalyzer::EPropertyType> m_vTypesForTypeActive;
	std::vector<CResultsAnalyzer::EPropertyType> m_vTypesForDistanceActive;
	std::vector<CResultsAnalyzer::EPropertyType> m_vTypesForComponentActive;
	std::vector<CResultsAnalyzer::EPropertyType> m_vTypesForDistrParamsActive;
	CAnalyzerThread *m_pAnalyzerThread;
	QTimer m_UpdateTimer;
	QSize m_size;

public:
	CResultsAnalyzerTab(QWidget *parent = 0);
	~CResultsAnalyzerTab() = 0;

	void SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB);
	void UpdateSettings();
	void Initialize();
	virtual void InitializeAnalyzerTab() {};

protected:
	// Visibility of GUI elements
	void SetResultsTypeVisible(bool _bVisible);
	void SetDistanceVisible(bool _bVisible);
	void SetComponentVisible(bool _bVisible);
	void SetCollisionsVisible(bool _bVisible);
	void SetGeometryVisible(bool _bVisible);
	void SetPoint2Visible(bool _bVisible);
	void SetConstraintsVisible(bool _bVisible);
	void SetConstraintMaterialsVisible(bool _bVisible);
	void SetConstraintMaterials2Visible(bool _bVisible);
	void SetConstraintVolumesVisible(bool _bVisible);
	void SetConstraintGeometriesVisible(bool _bVisible);
	void SetConstraintDiametersVisible(bool _bVisible);
	void SetConstraintDiameters2Visible(bool _bVisible);

	// Update controls according to CResultsAnalyzer
	void UpdateSelectedProperty();
	void UpdateSelectedResultsType();
	void UpdateSelectedDistance();
	void UpdateSelectedComponent();
	void UpdateSelectedRelation();
	void UpdateSelectedCollisionsType();
	void UpdateSelectedGeometry();
	void UpdateDistance();
	void UpdateTime();
	void UpdateDistrParams();
	void UpdateConstraints();

	// Update activity of controls
	void UpdateResultsTypeActivity();
	virtual void UpdateDistanceVisibility();
	void UpdateComponentActivity();
	void UpdateDistrParamsActivity();
	void UpdateTimeParams();

	void SetMultiplePropertySelection(bool _allow) const; // Sets the selection mode of the list widget and limits the height of the widget to approx 6 rows
	void SetWindowTitle(const QString& _sTitle);
	void SetupGeometryCombo();
	void SetStatusText(const QString& _sText);
	void SetComponentActive(bool _bActive);

private:
	void InitializeConnections() const;
	void SetDistrParamsActive(bool _bActive);

public slots:
	virtual void UpdateWholeView();

protected slots:
	void ExportDataPressed();
	virtual void ExportData();
	virtual void NewPropertySelected(int _nRow);
	void AddAnalysisProperty(CResultsAnalyzer::EPropertyType _property, const QString& _rowNameComboBox, const QString& _sToolTip);
	virtual void NewResultsTypeSelected(bool _bChecked);
	virtual void NewDistanceTypeSelected(bool _bChecked);
	virtual void NewComponentSelected(bool _bChecked);
	virtual void NewRelationSelected(bool _bChecked);
	virtual void NewCollisionsTypeSelected(bool _bChecked);
	virtual void NewGeometrySelected(int _nIndex);
	virtual void NewDataPointsSet();
	virtual void NewTimeSet();
	virtual void NewDistrParamSet();
	void setVisible(bool _bVisible);

private slots:
	void CloseDialog(int _nResult);
	void UpdateExportStatistics();
	void ExportFinished();
};
