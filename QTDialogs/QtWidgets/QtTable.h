/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "UnitConvertor.h"
#include "Quaternion.h"
#include "ColorView.h"
#include "TableItemSpinBox.h"
#include <QTableWidget>
#include <QKeyEvent>
#include <QCheckBox>
#include <QComboBox>
#include <QProgressBar>
#include <QPushButton>

class CMUSENTableItem : public QTableWidgetItem
{
public:
	CMUSENTableItem() : QTableWidgetItem() {}
	CMUSENTableItem(const QString& _text) : QTableWidgetItem{ _text } {}

	bool operator<(const QTableWidgetItem& _other) const override
	{
		if (text().toDouble() != 0 || _other.text().toDouble() != 0)
			return text().toDouble() < _other.text().toDouble();
		else
			return text() < _other.text();
	}
};

class CQtTable : public QTableWidget
{
	Q_OBJECT

public:
	struct SPasteInfo
	{
		int firstModifiedRow;
		int lastModifiedRow;
	};

private:
	const CUnitConvertor* m_unitConverter;
	bool m_bBlockingPaste;
	SPasteInfo m_pasteInfo;

public:
	CQtTable( QWidget *parent = 0 );
	CQtTable( int rows, int columns, QWidget * parent = 0 );

	void SetUnitConverter(const CUnitConvertor* _unitConverter);

	void BlockingPaste( bool _bBlock ){ m_bBlockingPaste = _bBlock; }
	SPasteInfo GetModifiedRows() const;

	void SetColHeaderItem(int _col, const std::string& _text);
	void SetRowHeaderItem(int _row, const std::string& _text);
	void SetColHeaderItems(int _startcol, const std::vector<std::string>& _text);
	void SetRowHeaderItems(int _startrow, const std::vector<std::string>& _text);
	void SetColHeaderItemConv(int _col, const std::string& _text, EUnitType _units = EUnitType::NONE);
	void SetRowHeaderItemConv(int _row, const std::string& _text, EUnitType _units = EUnitType::NONE);

	void SetItemEditable(int _row, int _col, const QString& _text, const QVariant& _userData = -1);
	void SetItemEditable(int _row, int _col, const std::string& _text, const QVariant& _userData = -1);
	void SetItemEditable(int _row, int _col, double _value, const QVariant& _userData = -1);
	void SetItemEditable(int _row, int _col, unsigned _value, const QVariant& _userData = -1);
	void SetItemEditable(int _row, int _col);
	void SetItemEditableConv(int _row, int _col, double _value, EUnitType _units = EUnitType::NONE);

	void SetItemsColEditable(int _startrow, int _col, const std::vector<double>& _val);
	void SetItemsColEditable(int _startrow, int _col, const std::vector<std::string>& _val);
	void SetItemsRowEditable(int _row, int _startcol, const std::vector<double>& _val);
	void SetItemsRowEditable(int _row, int _startcol, const std::vector<std::string>& _val);
	void SetItemsColEditableConv(int _startrow, int _col, const CVector3& _val, EUnitType _units = EUnitType::NONE);
	void SetItemsRowEditableConv(int _row, int _startcol, const CVector3& _val, EUnitType _units = EUnitType::NONE);
	void SetItemsColEditableConv(int _startrow, int _col, const CQuaternion& _val, EUnitType _units = EUnitType::NONE);
	void SetItemsRowEditableConv(int _row, int _startcol, const CQuaternion& _val, EUnitType _units = EUnitType::NONE);

	void SetItemNotEditable(int _row, int _col, const QString& _text, const QVariant& _userData = -1);
	void SetItemNotEditable(int _row, int _col, const std::string& _text, const QVariant& _userData = -1);
	void SetItemNotEditable(int _row, int _col, double _value, const QVariant& _userData = -1);
	void SetItemNotEditable(int _row, int _col, unsigned _value, const QVariant& _userData = -1);
	void SetItemNotEditable(int _row, int _col, size_t _value, const QVariant& _userData = -1);
	void SetItemNotEditable(int _row, int _col);
	void SetItemNotEditableConv(int _row, int _col, double _value, EUnitType _units);

	void SetItemsColNotEditable(int _startrow, int _col, const std::vector<double>& _val);
	void SetItemsColNotEditable(int _startrow, int _col, const std::vector<std::string>& _val);
	void SetItemsRowNotEditable(int _row, int _startcol, const std::vector<double>& _val);
	void SetItemsRowNotEditable(int _row, int _startcol, const std::vector<std::string>& _val);
	void SetItemsColNotEditableConv(int _startrow, int _col, const CVector3& _val, EUnitType _units = EUnitType::NONE);
	void SetItemsRowNotEditableConv(int _row, int _startcol, const CVector3& _val, EUnitType _units = EUnitType::NONE);

	double GetConvValue(int _row, int _col, EUnitType _units = EUnitType::NONE) const;
	CVector3 GetConvVectorRow(int _row, int _col, EUnitType _units = EUnitType::NONE) const;
	CVector3 GetConvVectorCol(int _row, int _col, EUnitType _units = EUnitType::NONE) const;
	CQuaternion GetConvQuartRow(int _row, int _col, EUnitType _units = EUnitType::NONE) const;
	CQuaternion GetConvQuartCol(int _row, int _col, EUnitType _units = EUnitType::NONE) const;

	QCheckBox* SetCheckBox(int _row, int _col, bool _checked = true);
	QCheckBox* GetCheckBox(int _row, int _col) const;
	void SetCheckBoxChecked(int _row, int _col, bool _checked) const;
	bool GetCheckBoxChecked(int _row, int _col) const;

	CTableItemSpinBox* SetDoubleSpinBox(int _row, int _col, double _value);
	CTableItemSpinBox* GetDoubleSpinBox(int _row, int _col) const;
	double GetDoubleSpinBoxValue(int _row, int _col) const;
	void SetDoubleSpinBoxConv(int _row, int _col, double _value, EUnitType _units = EUnitType::NONE);
	double GetDoubleSpinBoxConv(int _row, int _col, EUnitType _units = EUnitType::NONE) const;
	void SetDoubleSpinBoxColConv(int _row, int _col, const CVector3& _value, EUnitType _units = EUnitType::NONE);
	void SetDoubleSpinBoxRowConv(int _row, int _col, const CVector3& _value, EUnitType _units = EUnitType::NONE);
	CVector3 GetDoubleSpinBoxColConv(int _row, int _col, EUnitType _units = EUnitType::NONE) const;
	CVector3 GetDoubleSpinBoxRowConv(int _row, int _col, EUnitType _units = EUnitType::NONE) const;

	QComboBox* SetComboBox(int _row, int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, int _iSelected);
	QComboBox* SetComboBox(int _row, int _col, const std::vector<std::string>& _names, const std::vector<std::string>& _data, const std::string& _dataSelected);
	QComboBox* SetComboBox(int _row, int _col, const std::vector<std::string>& _names, const std::vector<size_t>& _data, size_t _dataSelected);
	QComboBox* GetComboBox(int _row, int _col) const;
	void SetComboBoxValue(int _row, int _col, const QVariant& _val) const;
	QVariant GetComboBoxValue(int _row, int _col) const;

	QPushButton* SetPushButton(int _row, int _col, const QString& _text);
	QPushButton* GetPushButton(int _row, int _col) const;

	QProgressBar* SetProgressBar(int _row, int _col, int _value);
	QProgressBar* GetProgressBar(int _row, int _col) const;

	QSlider* SetSlider(int _row, int _col, int _value);
	QSlider* GetSlider(int _row, int _col) const;
	int GetSliderValue(int _row, int _col) const;

	CColorView* SetColorPicker(int _row, int _col, const CColor& _color);
	CColorView* GetColorPicker(int _row, int _col) const;
	CColor GetColorPickerColor(int _row, int _col) const;

	void SetRowBackgroundColor(const QColor& _color, int _row) const;
	void SetColumnBackgroundColor(const QColor& _color, int _col) const;

	std::pair<int, int> CurrentCellPos() const;
	void RestoreSelectedCell(const std::pair<int, int>& _cellPos);
	void RestoreSelectedCell(int _row, int _col);

	void SetItemUserData(int _row, int _col, const QVariant& _data) const;
	QVariant GetItemUserData(int _row = -1, int _col = -1) const;

	void ShowRow(int _row, bool _show);
	void ShowCol(int _col, bool _show);

	void mousePressEvent(QMouseEvent* _event) override;

public slots:
	void keyPressEvent( QKeyEvent *event ) override;

private:
	void Clear();
	void Copy() const;
	void Paste();

	static void SetComboBoxValue(QComboBox* _combo, const QVariant& _val);

signals:
	void DataPasted();
	void CheckBoxStateChanged(int _row, int _col, QCheckBox* _checkBox);
	void ComboBoxIndexChanged(int _row, int _col, QComboBox* _comboBox);
	void PushButtonClicked(int _row, int _col, QPushButton* _pushButton);
	void EmptyAreaClicked();
};
