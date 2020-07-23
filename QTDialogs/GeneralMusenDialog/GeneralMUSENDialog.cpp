/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeneralMUSENDialog.h"
#include "qtOperations.h"
#include <QCoreApplication>
#include <QDesktopServices>
#include <QKeyEvent>
#include <QLineEdit>

CMusenDialog::CMusenDialog(QWidget *parent) : QDialog(parent)
{
	m_bAvoidSignal = false;
	m_pSystemStructure = nullptr;
	m_pUnitConverter = nullptr;
	m_pMaterialsDB = nullptr;
	m_pGeometriesDB = nullptr;
	m_pAgglomDB = nullptr;
	m_dCurrentTime = 0;
	m_sHelpFileName = "";

	// hide question symbol from all windows
	setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
}

void CMusenDialog::ShowDialog()
{
	show();
	raise();
}

void CMusenDialog::OpenHelpFile() const
{
	if (!m_sHelpFileName.isEmpty())
		QDesktopServices::openUrl(QUrl::fromLocalFile("file:///" + QCoreApplication::applicationDirPath() + "/Documentation/" + m_sHelpFileName));
}

void CMusenDialog::SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB,
	CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB)
{
	m_pSystemStructure = _pSystemStructure;
	m_pUnitConverter = _pUnitConvertor;
	m_pMaterialsDB = _pMaterialsDB;
	m_pGeometriesDB = _pGeometriesDB;
	m_pAgglomDB = _pAgglomDB;
	emit PointersAreSet();
}

void CMusenDialog::SetCurrentTime(double _dTime)
{
	m_dCurrentTime = _dTime;
	if (isVisible())
		UpdateWholeView();
}

void CMusenDialog::setVisible(bool _bVisible)
{
	QDialog::setVisible(_bVisible);
	if (_bVisible)
		UpdateWholeView();
}

void CMusenDialog::keyPressEvent(QKeyEvent* event)
{
	switch (event->key())
	{
	case Qt::Key_F1: OpenHelpFile();				break;
	default:		 QDialog::keyPressEvent(event);	break;
	}
}

void CMusenDialog::SetWindowModal(bool _modal)
{
	const QRect oldGeometry = geometry();
	hide();
	setWindowModality(_modal ? Qt::ApplicationModal : Qt::NonModal);
	show();
	setGeometry(oldGeometry);
}

void CMusenDialog::ShowConvLabel(QTableWidgetItem* _pItem, const QString& _sLabel, EUnitType _nUnitType) const
{
	_pItem->setText(_sLabel + " [" + ss2qs(m_pUnitConverter->GetSelectedUnit(_nUnitType)) + "]");
}

void CMusenDialog::ShowConvLabel(QLabel* _pItem, const QString& _sLabel, EUnitType _nUnitType) const
{
	_pItem->setText(_sLabel + " [" + ss2qs(m_pUnitConverter->GetSelectedUnit(_nUnitType)) + "]");
}

void CMusenDialog::ShowConvLabel(QRadioButton* _pItem, const QString& _sLabel, EUnitType _nUnitType) const
{
	_pItem->setText(_sLabel + " [" + ss2qs(m_pUnitConverter->GetSelectedUnit(_nUnitType)) + "]");
}

void CMusenDialog::ShowConvValue(QTableWidgetItem* _pItem, double _dValue, EUnitType _nUnitType) const
{
	_pItem->setText(QString::number(m_pUnitConverter->GetValue(_nUnitType, _dValue)));
}

void CMusenDialog::ShowConvValue(QTableWidgetItem* _pItem, double _dValue) const
{
	_pItem->setText(QString::number(_dValue));
}

void CMusenDialog::ShowConvValue(QLabel* _pItem, double _dValue, EUnitType _nUnitType) const
{
	_pItem->setText(QString::number(m_pUnitConverter->GetValue(_nUnitType, _dValue)));
}

void CMusenDialog::ShowConvValue(QLineEdit* _pItem, double _dValue, EUnitType _nUnitType, int _precision/* = -1*/) const
{
	if (_precision == -1)
		_pItem->setText(QString::number(m_pUnitConverter->GetValue(_nUnitType, _dValue)));
	else
		_pItem->setText(QString::number(m_pUnitConverter->GetValue(_nUnitType, _dValue), 'g', _precision));
}

void CMusenDialog::ShowConvValue(QLineEdit* _pL1, QLineEdit* _pL2, QLineEdit* _pL3, const CVector3& _vec, EUnitType _nUnitType) const
{
	ShowConvValue(_pL1, _vec.x, _nUnitType);
	ShowConvValue(_pL2, _vec.y, _nUnitType);
	ShowConvValue(_pL3, _vec.z, _nUnitType);
}

double CMusenDialog::GetConvValue(const QLineEdit* _pItem, EUnitType _nUnitType) const
{
	return m_pUnitConverter->GetValueSI(_nUnitType, _pItem->text().toDouble());
}

double CMusenDialog::GetConvValue(const QTableWidgetItem* _pItem, EUnitType _nUnitType) const
{
	return m_pUnitConverter->GetValueSI(_nUnitType, _pItem->text().toDouble());
}

void CMusenDialog::ShowVectorInTableRow(const CVector3& _vVec, QTableWidget* _pTable, int _nRow, int _nStartColumn, EUnitType _nDataType /*= 0 */) const
{
	if (_nRow >= _pTable->rowCount()) return;
	if (_nStartColumn + 2 >= _pTable->columnCount()) return;
	ShowConvValue(_pTable->item(_nRow, _nStartColumn), _vVec.x, _nDataType);
	ShowConvValue(_pTable->item(_nRow, _nStartColumn + 1), _vVec.y, _nDataType);
	ShowConvValue(_pTable->item(_nRow, _nStartColumn + 2), _vVec.z, _nDataType);
}

CVector3 CMusenDialog::GetVectorFromTableRow(QTableWidget* _pTable, int _nRow, int _nStartColumn, EUnitType _nDataType) const
{
	CVector3 vResult{ 0, 0, 0 };
	if (_nRow >= _pTable->rowCount()) return vResult;
	if (_nStartColumn + 2 >= _pTable->columnCount()) return vResult;
	vResult.x = GetConvValue(_pTable->item(_nRow, _nStartColumn), _nDataType);
	vResult.y = GetConvValue(_pTable->item(_nRow, _nStartColumn + 1), _nDataType);
	vResult.z = GetConvValue(_pTable->item(_nRow, _nStartColumn + 2), _nDataType);
	return vResult;
}

void CMusenDialog::ShowVectorInTableColumn(const CVector3& _vVec, QTableWidget* _pTable, int _nStartRow, int _nColumn, EUnitType _nDataType) const
{
	if (_nColumn >= _pTable->columnCount()) return;
	if (_nStartRow + 2 >= _pTable->rowCount()) return;
	ShowConvValue(_pTable->item(_nStartRow, _nColumn), _vVec.x, _nDataType);
	ShowConvValue(_pTable->item(_nStartRow + 1, _nColumn), _vVec.y, _nDataType);
	ShowConvValue(_pTable->item(_nStartRow + 2, _nColumn), _vVec.z, _nDataType);
}

CVector3 CMusenDialog::GetVectorFromTableColumn(QTableWidget* _pTable, int _nStartRow, int _nColumn, EUnitType _nDataType) const
{
	CVector3 vResult{ 0, 0, 0 };
	if (_nColumn >= _pTable->columnCount()) return vResult;
	if (_nStartRow + 2 >= _pTable->rowCount()) return vResult;
	vResult.x = GetConvValue(_pTable->item(_nStartRow, _nColumn), _nDataType);
	vResult.y = GetConvValue(_pTable->item(_nStartRow + 1, _nColumn), _nDataType);
	vResult.z = GetConvValue(_pTable->item(_nStartRow + 2, _nColumn), _nDataType);
	return vResult;
}

CVector3 CMusenDialog::GetConvValue(const QLineEdit* _pL1, const QLineEdit* _pL2, const QLineEdit* _pL3, EUnitType _nDataType /*= 0*/) const
{
	return CVector3(GetConvValue(_pL1, _nDataType), GetConvValue(_pL2, _nDataType), GetConvValue(_pL3, _nDataType));
}

bool CMusenDialog::IsFileWritable(const QString& _sFilePath)
{
	QFile tempFile(_sFilePath);
	const bool bIsWritable = tempFile.open(QIODevice::ReadWrite);
	tempFile.close();
	return bIsWritable;
}
