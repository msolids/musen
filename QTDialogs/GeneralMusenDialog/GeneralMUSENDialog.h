/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"
#include "UnitConvertor.h"
#include "GeometriesDatabase.h"
#include "AgglomeratesDatabase.h"
#include "SimulatorManager.h"
#include <QDialog>
#include <QTableWidgetItem>
#include <QLabel>
#include <QRadioButton>

class CMusenDialog : public QDialog
{
	Q_OBJECT
protected:
	QString m_sHelpFileName; // name of the file where documentation is stored

	CSystemStructure* m_pSystemStructure;
	CUnitConvertor* m_pUnitConverter;
	CMaterialsDatabase* m_pMaterialsDB;
	CGeometriesDatabase* m_pGeometriesDB;
	CAgglomeratesDatabase* m_pAgglomDB;
	bool m_bAvoidSignal;

	double m_dCurrentTime;

public slots:
	void ShowDialog();
	virtual void UpdateWholeView() {};
	void setVisible(bool _bVisible) override;
	virtual void NewSceneLoaded() {}

signals:
	void UpdateOpenGLView();
	void EnableOpenGLView();
	void DisableOpenGLView();
	void PointersAreSet();
	void UpdateViewParticles();
	void UpdateViewBonds();
	void UpdateViewGeometries();
	void UpdateViewVolumes();
	void UpdateViewSlices();
	void UpdateViewDomain();
	void UpdateViewPBC();
	void UpdateViewAxes();
	void UpdateViewTime();
	void UpdateViewLegend();

protected:
	void OpenHelpFile() const;
	void keyPressEvent(QKeyEvent *event) override;
	void SetWindowModal(bool _modal);

	// set of functions to work with units converter
	void ShowConvLabel(QTableWidgetItem* _pItem, const QString& _sLabel, EUnitType _nUnitType) const;
	void ShowConvLabel(QLabel* _pItem, const QString& _sLabel, EUnitType _nUnitType) const;
	void ShowConvLabel(QRadioButton* _pItem, const QString& _sLabel, EUnitType _nUnitType) const;
	void ShowConvValue(QTableWidgetItem* _pItem, double _dValue) const;
	void ShowConvValue(QTableWidgetItem* _pItem, double _dValue, EUnitType _nUnitType) const;
	void ShowConvValue(QLineEdit* _pItem, double _dValue, EUnitType _nUnitType, int _precision = -1) const;
	void ShowConvValue(QLabel* _pItem, double _dValue, EUnitType _nUnitType) const;
	void ShowConvValue(QLineEdit* _pL1, QLineEdit* _pL2, QLineEdit* _pL3, const CVector3& _vec, EUnitType _nUnitType) const;


	double GetConvValue(const QLineEdit* _pItem, EUnitType _nUnitType) const;
	double GetConvValue(const QTableWidgetItem* _pItem, EUnitType _nUnitType) const;
	CVector3 GetConvValue(const QLineEdit* _pL1, const QLineEdit* _pL2, const QLineEdit* _pL3, EUnitType _nDataType) const;

	void ShowVectorInTableRow(const CVector3& _vVec, QTableWidget* _pTable, int _nRow, int _nStartColumn, EUnitType _nDataType = EUnitType::NONE) const;
	CVector3 GetVectorFromTableRow(QTableWidget* _pTable, int _nRow, int _nStartColumn, EUnitType _nDataType = EUnitType::NONE) const;
	void ShowVectorInTableColumn(const CVector3& _vVec, QTableWidget* _pTable, int _nStartRow, int _nColumn, EUnitType _nDataType = EUnitType::NONE) const;
	CVector3 GetVectorFromTableColumn(QTableWidget* _pTable, int _nStartRow, int _nColumn, EUnitType _nDataType = EUnitType::NONE) const;

public:
	CMusenDialog(QWidget *parent);

	virtual void SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor,
		CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB);

	void SetCurrentTime(double _dTime);
	static bool IsFileWritable(const QString& _sFilePath);

	virtual void Initialize() {};
};
