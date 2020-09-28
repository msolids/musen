/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "QtTable.h"
#include <QApplication>
#include <QHBoxLayout>
#include <QClipboard>

CQtTable::CQtTable( QWidget *parent )
	: QTableWidget( parent )
{
	m_bBlockingPaste = false;
	m_pasteInfo.firstModifiedRow = m_pasteInfo.lastModifiedRow = 0;
}

CQtTable::CQtTable( int rows, int columns, QWidget *parent )
	: QTableWidget( rows, columns, parent )
{
	m_bBlockingPaste = false;
	m_pasteInfo.firstModifiedRow = m_pasteInfo.lastModifiedRow = 0;
}

void CQtTable::SetUnitConverter(const CUnitConvertor* _unitConverter)
{
	m_unitConverter = _unitConverter;
}

void CQtTable::Clear()
{
	const bool block = blockSignals(m_bBlockingPaste);

	QModelIndexList indexes = selectionModel()->selection().indexes();
	m_pasteInfo.firstModifiedRow = indexes.count() ? indexes.front().row() : 0;
	m_pasteInfo.lastModifiedRow  = 0;

	for (auto& index : indexes)
	{
		if (isRowHidden(index.row()) || isColumnHidden(index.column())) continue;

		if (auto* item = this->item(index.row(), index.column()))
			item->setText("");
		else if (auto* combo = GetComboBox(index.row(), index.column()))
			combo->setCurrentIndex(-1);
		else if (auto* checkBox = GetCheckBox(index.row(), index.column()))
			checkBox->setChecked(false);
		else if (auto* picker = GetColorPicker(index.row(), index.column()))
			picker->SetColor(Qt::lightGray);
		else if (auto* spinBox = GetDoubleSpinBox(index.row(), index.column()))
			spinBox->SetValue(0.0);
		else if (auto* slider = GetSlider(index.row(), index.column()))
			slider->setValue(0.0);

		if (m_bBlockingPaste)
		{
			if (index.row() < m_pasteInfo.firstModifiedRow)
				m_pasteInfo.firstModifiedRow = index.row();
			if (index.row() > m_pasteInfo.lastModifiedRow)
				m_pasteInfo.lastModifiedRow = index.row();
		}
	}

	blockSignals(block);
	emit DataPasted();
}

void CQtTable::Copy() const
{
	const QModelIndexList indexes = selectionModel()->selection().indexes();
	if (indexes.empty()) return;
	QString str;
	for (int iRow = indexes.front().row(); iRow <= indexes.back().row(); ++iRow)
	{
		if (isRowHidden(iRow)) continue;
		for (int iCol = indexes.front().column(); iCol <= indexes.back().column(); ++iCol)
		{
			if (isColumnHidden(iCol)) continue;
			if (const auto* item = this->item(iRow, iCol))
				str += item->text();
			else if (const auto* combo = GetComboBox(iRow, iCol))
				str += combo->itemData(combo->currentIndex()).toString();
			else if (const auto* checkBox = GetCheckBox(iRow, iCol))
				str += QString::number(checkBox->isChecked());
			else if (const auto* picker = GetColorPicker(iRow, iCol))
				str += picker->getColor().name();
			else if (const auto* spinBox = GetDoubleSpinBox(iRow, iCol))
				str += QString::number(spinBox->GetValue());
			else if (const auto* slider = GetSlider(iRow, iCol))
				str += QString::number(slider->value());
			if (iCol != indexes.back().column())
				str += "\t";
		}
		str += "\n";
	}
	if (!str.isEmpty()) // remove last EOL
		str.chop(1);
	QApplication::clipboard()->setText(str);
}

void CQtTable::Paste()
{
	const bool block = blockSignals(m_bBlockingPaste);
	const QString selectedText = QApplication::clipboard()->text();
	QStringList rows = selectedText.split(QRegExp(QLatin1String("\n")));
	while (!rows.isEmpty() && rows.back().isEmpty())
		rows.pop_back();

	const QModelIndexList indexes = selectionModel()->selection().indexes();
	const int firstRow = indexes.count() ? indexes.at(0).row()    : 0;
	const int firstCol = indexes.count() ? indexes.at(0).column() : 0;

	const int maxRow = std::min(rows.count(), rowCount() - firstRow);
	int rowsHidden = 0;
	for (int iRow = 0; iRow < maxRow; ++iRow)
	{
		for (int i = iRow + firstRow + rowsHidden; i < rowCount() && isRowHidden(i); ++i)
			rowsHidden++;
		const int iRealRow = iRow + firstRow + rowsHidden;

		QStringList columns = rows[iRow].split(QRegExp("[\t ]"));
		while (!columns.isEmpty() && columns.back().isEmpty())
			columns.pop_back();
		const int maxCol = std::min(columns.count(), columnCount() - firstCol);
		int colsHidden = 0;
		for (int iCol = 0; iCol < maxCol; ++iCol)
		{
			for (int i = iCol + firstCol + colsHidden; i < columnCount() && isColumnHidden(i); ++i)
				colsHidden++;
			const int iRealCol = iCol + firstCol + colsHidden;

			if (auto* item = this->item(iRealRow, iRealCol))
			{
				if (item->flags() & Qt::ItemIsEditable)
					item->setText(QString::number(columns[iCol].toDouble()));
			}
			else if (auto* combo = GetComboBox(iRealRow, iRealCol))
				SetComboBoxValue(combo, columns[iCol]);
			else if (auto* checkBox = GetCheckBox(iRealRow, iRealCol))
				checkBox->setChecked(columns[iCol].toInt());
			else if (auto* picker = GetColorPicker(iRealRow, iRealCol))
				picker->SetColor(QColor{ columns[iCol] });
			else if (auto* spinBox = GetDoubleSpinBox(iRealRow, iRealCol))
				spinBox->SetValue(columns[iCol].toDouble());
			else if (auto* slider = GetSlider(iRealRow, iRealCol))
				slider->setValue(columns[iCol].toInt());
		}
	}
	m_pasteInfo.firstModifiedRow = firstRow;
	m_pasteInfo.lastModifiedRow  = firstRow + maxRow;
	blockSignals(block);
	emit DataPasted();
}

CQtTable::SPasteInfo CQtTable::GetModifiedRows() const
{
	return m_pasteInfo;
}

void CQtTable::SetColHeaderItem(int _col, const std::string& _text)
{
	if (horizontalHeaderItem(_col))
		horizontalHeaderItem(_col)->setText(QString::fromStdString(_text));
	else
		setHorizontalHeaderItem(_col, new QTableWidgetItem(QString::fromStdString(_text)));
}

void CQtTable::SetRowHeaderItem(int _row, const std::string& _text)
{
	if (verticalHeaderItem(_row))
		verticalHeaderItem(_row)->setText(QString::fromStdString(_text));
	else
		setVerticalHeaderItem(_row, new QTableWidgetItem(QString::fromStdString(_text)));
}

void CQtTable::SetColHeaderItems(int _startcol, const std::vector<std::string>& _text)
{
	for (int i = 0; i < static_cast<int>(_text.size()); ++i)
		if (_startcol + i < columnCount())
			SetColHeaderItem(_startcol + i, _text[i]);
}

void CQtTable::SetRowHeaderItems(int _startrow, const std::vector<std::string>& _text)
{
	for (int i = 0; i < static_cast<int>(_text.size()); ++i)
		if (_startrow + i < rowCount())
			SetRowHeaderItem(_startrow + i, _text[i]);
}

void CQtTable::SetColHeaderItemConv(int _col, const std::string& _text, EUnitType _units)
{
	SetColHeaderItem(_col, _text + (_units != EUnitType::NONE ? " [" + m_unitConverter->GetSelectedUnit(_units) + "]" : ""));
}

void CQtTable::SetRowHeaderItemConv(int _row, const std::string& _text, EUnitType _units)
{
	SetRowHeaderItem(_row, _text + (_units != EUnitType::NONE ? " [" + m_unitConverter->GetSelectedUnit(_units) + "]" : ""));
}

void CQtTable::SetItemEditable(int _row, int _col, const QString& _text, const QVariant& _userData /*= -1*/)
{
	SetItemNotEditable(_row, _col, _text, _userData);
	item(_row, _col)->setFlags(item(_row, _col)->flags() | Qt::ItemIsEditable);
}

void CQtTable::SetItemEditable(int _row, int _col, const std::string& _text, const QVariant& _userData)
{
	SetItemEditable(_row, _col, QString::fromStdString(_text), _userData);
}

void CQtTable::SetItemEditable(int _row, int _col, double _value, const QVariant& _userData)
{
	SetItemEditable(_row, _col, QString::number(_value), _userData);
}

void CQtTable::SetItemEditable(int _row, int _col, unsigned _value, const QVariant& _userData)
{
	SetItemEditable(_row, _col, QString::number(_value), _userData);
}

void CQtTable::SetItemEditable(int _row, int _col)
{
	SetItemEditable(_row, _col, QString{});
}

void CQtTable::SetItemEditableConv(int _row, int _col, double _value, EUnitType _units)
{
	SetItemEditable(_row, _col, QString::number(m_unitConverter->GetValue(_units, _value)));
}

void CQtTable::SetItemsColEditable(int _startrow, int _col, const std::vector<double>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startrow + i < rowCount())
			SetItemEditable(_startrow + i, _col, _val[i]);
}

void CQtTable::SetItemsColEditable(int _startrow, int _col, const std::vector<std::string>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startrow + i < rowCount())
			SetItemEditable(_startrow + i, _col, _val[i]);
}

void CQtTable::SetItemsRowEditable(int _row, int _startcol, const std::vector<double>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startcol + i < columnCount())
			SetItemEditable(_row, _startcol + i, _val[i]);
}

void CQtTable::SetItemsRowEditable(int _row, int _startcol, const std::vector<std::string>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startcol + i < columnCount())
			SetItemEditable(_row, _startcol + i, _val[i]);
}

void CQtTable::SetItemsColEditableConv(int _startrow, int _col, const CVector3& _val, EUnitType _units)
{
	if (_startrow + 2 >= rowCount()) return;
	SetItemEditableConv(_startrow + 0, _col, _val.x, _units);
	SetItemEditableConv(_startrow + 1, _col, _val.y, _units);
	SetItemEditableConv(_startrow + 2, _col, _val.z, _units);
}

void CQtTable::SetItemsRowEditableConv(int _row, int _startcol, const CVector3& _val, EUnitType _units)
{
	if (_startcol + 2 >= columnCount()) return;
	SetItemEditableConv(_row, _startcol + 0, _val.x, _units);
	SetItemEditableConv(_row, _startcol + 1, _val.y, _units);
	SetItemEditableConv(_row, _startcol + 2, _val.z, _units);
}

void CQtTable::SetItemsColEditableConv(int _startrow, int _col, const CQuaternion& _val, EUnitType _units)
{
	if (_startrow + 3 >= rowCount()) return;
	SetItemEditableConv(_startrow + 0, _col, _val.q0, _units);
	SetItemEditableConv(_startrow + 1, _col, _val.q1, _units);
	SetItemEditableConv(_startrow + 2, _col, _val.q2, _units);
	SetItemEditableConv(_startrow + 3, _col, _val.q3, _units);
}

void CQtTable::SetItemsRowEditableConv(int _row, int _startcol, const CQuaternion& _val, EUnitType _units)
{
	if (_startcol + 3 >= columnCount()) return;
	SetItemEditableConv(_row, _startcol + 0, _val.q0, _units);
	SetItemEditableConv(_row, _startcol + 1, _val.q1, _units);
	SetItemEditableConv(_row, _startcol + 2, _val.q2, _units);
	SetItemEditableConv(_row, _startcol + 3, _val.q3, _units);
}

void CQtTable::SetItemNotEditable(int _row, int _col, const QString& _text, const QVariant& _userData /*= -1*/)
{
	const bool exist = this->item(_row, _col) != nullptr;
	auto* item = exist ? this->item(_row, _col) : new CMUSENTableItem(_text);
	if (exist)
		item->setText(_text);
	item->setData(Qt::DisplayRole, _text);
	if (_userData != -1)
		item->setData(Qt::UserRole, _userData);
	item->setFlags(item->flags() & ~Qt::ItemIsEditable);
	if (!exist)
		setItem(_row, _col, item);
}

void CQtTable::SetItemNotEditable(int _row, int _col, const std::string& _text, const QVariant& _userData)
{
	SetItemNotEditable(_row, _col, QString::fromStdString(_text), _userData);
}

void CQtTable::SetItemNotEditable(int _row, int _col, double _value, const QVariant& _userData)
{
	SetItemNotEditable(_row, _col, QString::number(_value), _userData);
}

void CQtTable::SetItemNotEditable(int _row, int _col, unsigned _value, const QVariant& _userData)
{
	SetItemNotEditable(_row, _col, QString::number(_value), _userData);
}

void CQtTable::SetItemNotEditable(int _row, int _col, size_t _value, const QVariant& _userData)
{
	SetItemNotEditable(_row, _col, QString::number(_value), _userData);
}

void CQtTable::SetItemNotEditable(int _row, int _col)
{
	SetItemNotEditable(_row, _col, QString{});
}

void CQtTable::SetItemNotEditableConv(int _row, int _col, double _value, EUnitType _units)
{
	SetItemNotEditable(_row, _col, QString::number(m_unitConverter->GetValue(_units, _value)));
}

void CQtTable::SetItemsColNotEditable(int _startrow, int _col, const std::vector<double>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startrow + i < rowCount())
			SetItemNotEditable(_startrow + i, _col, _val[i]);
}

void CQtTable::SetItemsColNotEditable(int _startrow, int _col, const std::vector<std::string>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startrow + i < rowCount())
			SetItemNotEditable(_startrow + i, _col, _val[i]);
}

void CQtTable::SetItemsRowNotEditable(int _row, int _startcol, const std::vector<double>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startcol + i < columnCount())
			SetItemNotEditable(_row, _startcol + i, _val[i]);
}

void CQtTable::SetItemsRowNotEditable(int _row, int _startcol, const std::vector<std::string>& _val)
{
	for (int i = 0; i < static_cast<int>(_val.size()); ++i)
		if (_startcol + i < columnCount())
			SetItemNotEditable(_row, _startcol + i, _val[i]);
}

void CQtTable::SetItemsColNotEditableConv(int _startrow, int _col, const CVector3& _val, EUnitType _units)
{
	if (_startrow + 2 >= rowCount()) return;
	SetItemNotEditableConv(_startrow + 0, _col, _val.x, _units);
	SetItemNotEditableConv(_startrow + 1, _col, _val.y, _units);
	SetItemNotEditableConv(_startrow + 2, _col, _val.z, _units);
}

void CQtTable::SetItemsRowNotEditableConv(int _row, int _startcol, const CVector3& _val, EUnitType _units)
{
	if (_startcol + 2 >= columnCount()) return;
	SetItemNotEditableConv(_row, _startcol + 0, _val.x, _units);
	SetItemNotEditableConv(_row, _startcol + 1, _val.y, _units);
	SetItemNotEditableConv(_row, _startcol + 2, _val.z, _units);
}

double CQtTable::GetConvValue(int _row, int _col, EUnitType _units) const
{
	if (QTableWidgetItem* item = this->item(_row, _col))
		return m_unitConverter->GetValueSI(_units, item->text().toDouble());
	return {};
}

CVector3 CQtTable::GetConvVectorRow(int _row, int _col, EUnitType _units) const
{
	return CVector3{
		GetConvValue(_row, _col + 0, _units),
		GetConvValue(_row, _col + 1, _units),
		GetConvValue(_row, _col + 2, _units) };
}

CVector3 CQtTable::GetConvVectorCol(int _row, int _col, EUnitType _units) const
{
	return CVector3{
		GetConvValue(_row + 0, _col, _units),
		GetConvValue(_row + 1, _col, _units),
		GetConvValue(_row + 2, _col, _units) };
}

CQuaternion CQtTable::GetConvQuartRow(int _row, int _col, EUnitType _units) const
{
	return CQuaternion{
		GetConvValue(_row, _col + 0, _units),
		GetConvValue(_row, _col + 1, _units),
		GetConvValue(_row, _col + 2, _units),
		GetConvValue(_row, _col + 3, _units) };
}

CQuaternion CQtTable::GetConvQuartCol(int _row, int _col, EUnitType _units) const
{
	return CQuaternion{
		GetConvValue(_row + 0, _col, _units),
		GetConvValue(_row + 1, _col, _units),
		GetConvValue(_row + 2, _col, _units),
		GetConvValue(_row + 3, _col, _units) };
}

QCheckBox* CQtTable::SetCheckBox(const int _row, const int _col, bool _checked /*= true*/)
{
	auto* checkBox = GetCheckBox(_row, _col);
	if (checkBox)
		checkBox->setChecked(_checked);
	else
	{
		delete item(_row, _col);
		auto *widget = new QWidget(this);
		auto *layout = new QHBoxLayout(widget);
		checkBox = new QCheckBox(widget);
		layout->addWidget(checkBox);
		layout->setAlignment(Qt::AlignCenter);
		layout->setContentsMargins(0, 0, 0, 0);
		widget->setLayout(layout);
		checkBox->setChecked(_checked);
		checkBox->setObjectName("CheckBox");
		connect(checkBox, &QCheckBox::stateChanged, this, [=] { CheckBoxStateChanged(_row, _col, checkBox); });
		connect(checkBox, &QCheckBox::stateChanged, this, [=] { setCurrentCell(_row, _col); });
		setCellWidget(_row, _col, widget);
	}
	return checkBox;
}

QCheckBox* CQtTable::GetCheckBox(int _row, int _col) const
{
	return cellWidget(_row, _col)->findChild<QCheckBox*>("CheckBox");
}

void CQtTable::SetCheckBoxChecked(int _row, int _col, bool _checked) const
{
	auto* checkBox = GetCheckBox(_row, _col);
	if (!checkBox) return;
	QSignalBlocker blocker{ checkBox };
	checkBox->setChecked(_checked);
}

bool CQtTable::GetCheckBoxChecked(int _row, int _col) const
{
	auto* checkBox = GetCheckBox(_row, _col);
	if (!checkBox) return false;
	return checkBox->isChecked();
}

CTableItemSpinBox* CQtTable::SetDoubleSpinBox(int _row, int _col, double _value)
{
	auto* spinBox = GetDoubleSpinBox(_row, _col);
	if (spinBox)
	{
		QSignalBlocker blocker{ spinBox };
		spinBox->SetValue(_value);
	}
	else
	{
		delete item(_row, _col);
		spinBox = new CTableItemSpinBox(_value, this);
		setCellWidget(_row, _col, spinBox);
	}
	return spinBox;
}

CTableItemSpinBox* CQtTable::GetDoubleSpinBox(int _row, int _col) const
{
	return dynamic_cast<CTableItemSpinBox*>(cellWidget(_row, _col));
}

double CQtTable::GetDoubleSpinBoxValue(int _row, int _col) const
{
	auto* spinBox = GetDoubleSpinBox(_row, _col);
	if (!spinBox) return 0.0;
	return spinBox->GetValue();
}

void CQtTable::SetDoubleSpinBoxConv(int _row, int _col, double _value, EUnitType _units)
{
	SetDoubleSpinBox(_row, _col, m_unitConverter->GetValue(_units, _value));
}

double CQtTable::GetDoubleSpinBoxConv(int _row, int _col, EUnitType _units) const
{
	if (auto* spinBox = GetDoubleSpinBox(_row, _col))
		return m_unitConverter->GetValueSI(_units, spinBox->GetValue());
	return {};
}

void CQtTable::SetDoubleSpinBoxColConv(int _row, int _col, const CVector3& _value, EUnitType _units)
{
	if (_row + 2 >= rowCount()) return;
	SetDoubleSpinBoxConv(_row + 0, _col, _value.x, _units);
	SetDoubleSpinBoxConv(_row + 1, _col, _value.y, _units);
	SetDoubleSpinBoxConv(_row + 2, _col, _value.z, _units);
}

void CQtTable::SetDoubleSpinBoxRowConv(int _row, int _col, const CVector3& _value, EUnitType _units)
{
	if (_col + 2 >= columnCount()) return;
	SetDoubleSpinBoxConv(_row, _col + 0, _value.x, _units);
	SetDoubleSpinBoxConv(_row, _col + 1, _value.y, _units);
	SetDoubleSpinBoxConv(_row, _col + 2, _value.z, _units);
}

CVector3 CQtTable::GetDoubleSpinBoxColConv(int _row, int _col, EUnitType _units) const
{
	return CVector3{
		GetDoubleSpinBoxConv(_row + 0, _col, _units),
		GetDoubleSpinBoxConv(_row + 1, _col, _units),
		GetDoubleSpinBoxConv(_row + 2, _col, _units) };
}

CVector3 CQtTable::GetDoubleSpinBoxRowConv(int _row, int _col, EUnitType _units) const
{
	return CVector3{
		GetDoubleSpinBoxConv(_row, _col + 0, _units),
		GetDoubleSpinBoxConv(_row, _col + 2, _units),
		GetDoubleSpinBoxConv(_row, _col + 1, _units) };
}

QComboBox* CQtTable::SetComboBox(const int _row, const int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, int _iSelected)
{
	const auto SetupComboBox = [](QComboBox* _combo, const std::vector<QString>& _t, const std::vector<QVariant>& _d, int _i)
	{
		for (size_t i = 0; i < _t.size(); ++i)
			_combo->insertItem(_combo->count(), _t[i], i < _d.size() ? _d[i] : QVariant{});
		_combo->setCurrentIndex(_i);
	};

	auto* comboBox = GetComboBox(_row, _col);
	if (comboBox)
	{
		QSignalBlocker blocker{ comboBox };
		comboBox->clear();
		SetupComboBox(comboBox, _names, _data, _iSelected);
	}
	else
	{
		delete item(_row, _col);
		comboBox = new QComboBox(this);
		SetupComboBox(comboBox, _names, _data, _iSelected);
		connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [=] { ComboBoxIndexChanged(_row, _col, comboBox); });
		connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [=] { setCurrentCell(_row, _col); });
		setCellWidget(_row, _col, comboBox);
	}

	return comboBox;
}

QComboBox* CQtTable::SetComboBox(int _row, int _col, const std::vector<std::string>& _names, const std::vector<std::string>& _data, const std::string& _dataSelected)
{
	if (_names.size() != _data.size()) return nullptr;
	std::vector<QString> names;
	std::vector<QVariant> data;
	int iSelected = -1;
	for (int i = 0; i < static_cast<int>(_names.size()); ++i)
	{
		names.emplace_back(QString::fromStdString(_names[i]));
		data.emplace_back(QString::fromStdString(_data[i]));
		if (_data[i] == _dataSelected)	// this one is selected
			iSelected = i;
	}
	return SetComboBox(_row, _col, names, data, iSelected);
}

QComboBox* CQtTable::SetComboBox(int _row, int _col, const std::vector<std::string>& _names, const std::vector<size_t>& _data, size_t _dataSelected)
{
	if (_names.size() != _data.size()) return nullptr;
	std::vector<QString> names;
	std::vector<QVariant> data;
	int iSelected = -1;
	for (int i = 0; i < static_cast<int>(_names.size()); ++i)
	{
		names.emplace_back(QString::fromStdString(_names[i]));
		data.emplace_back(QVariant::fromValue(_data[i]));
		if (_data[i] == _dataSelected)	// this one is selected
			iSelected = i;
	}
	return SetComboBox(_row, _col, names, data, iSelected);
}

QComboBox* CQtTable::GetComboBox(int _row, int _col) const
{
	return dynamic_cast<QComboBox*>(cellWidget(_row, _col));
}

void CQtTable::SetComboBoxValue(int _row, int _col, const QVariant& _val) const
{
	auto* combo = GetComboBox(_row, _col);
	if (!combo) return;
	QSignalBlocker blocker{ combo };
	SetComboBoxValue(combo, _val);
}

void CQtTable::SetComboBoxValue(QComboBox* _combo, const QVariant& _val)
{
	for (int i = 0; i < _combo->count(); ++i)
		if (_combo->itemData(i) == _val)
		{
			_combo->setCurrentIndex(i);
			return;
		}
	_combo->setCurrentIndex(-1);
}

QVariant CQtTable::GetComboBoxValue(int _row, int _col) const
{
	auto* combo = GetComboBox(_row, _col);
	if (!combo) return{};
	return combo->itemData(combo->currentIndex());
}

QPushButton* CQtTable::SetPushButton(int _row, int _col, const QString& _text)
{
	delete item(_row, _col);
	auto* widget = new QWidget(this);
	auto* button = new QPushButton(widget);
	auto* layout = new QHBoxLayout(widget);
	layout->addWidget(button);
	layout->setAlignment(Qt::AlignCenter);
	layout->setContentsMargins(0, 0, 0, 0);
	widget->setLayout(layout);
	button->setText(_text);
	button->setAutoDefault(false);
	button->setObjectName("PushButton");
	connect(button, &QPushButton::clicked, this, [=] { PushButtonClicked(_row, _col, button); });
	setCellWidget(_row, _col, widget);
	return button;
}

QPushButton* CQtTable::GetPushButton(int _row, int _col) const
{
	return cellWidget(_row, _col)->findChild<QPushButton*>("PushButton");
}

QProgressBar* CQtTable::SetProgressBar(int _row, int _col, int _value)
{
	auto* bar = GetProgressBar(_row, _col);
	if (bar)
	{
		QSignalBlocker blocker{ bar };
		bar->setValue(_value);
	}
	else
	{
		delete item(_row, _col);
		bar = new QProgressBar{ this };
		bar->setRange(0, 100);
		bar->setValue(_value);
		setCellWidget(_row, _col, bar);
	}
	return bar;
}

QProgressBar* CQtTable::GetProgressBar(int _row, int _col) const
{
	return dynamic_cast<QProgressBar*>(cellWidget(_row, _col));
}

QSlider* CQtTable::SetSlider(int _row, int _col, int _value)
{
	auto* slider = GetSlider(_row, _col);
	if (slider)
	{
		QSignalBlocker blocker{ slider };
		slider->setValue(_value);
	}
	else
	{
		delete item(_row, _col);
		slider = new QSlider{ this };
		slider->setOrientation(Qt::Horizontal);
		slider->setTickPosition(QSlider::TicksBothSides);
		slider->setValue(_value);
		setCellWidget(_row, _col, slider);
	}
	return slider;
}

QSlider* CQtTable::GetSlider(int _row, int _col) const
{
	return dynamic_cast<QSlider*>(cellWidget(_row, _col));
}

int CQtTable::GetSliderValue(int _row, int _col) const
{
	auto* slider = GetSlider(_row, _col);
	if (!slider) return {};
	return slider->value();
}

CColorView* CQtTable::SetColorPicker(int _row, int _col, const CColor& _color)
{
	auto* picker = GetColorPicker(_row, _col);
	if (picker)
	{
		QSignalBlocker blocker{ picker };
		picker->SetColor(_color);
	}
	else
	{
		delete item(_row, _col);
		picker = new CColorView(this);
		picker->SetColor(_color);
		setCellWidget(_row, _col, picker);
	}
	return picker;
}

CColorView* CQtTable::GetColorPicker(int _row, int _col) const
{
	return dynamic_cast<CColorView*>(cellWidget(_row, _col));
}

CColor CQtTable::GetColorPickerColor(int _row, int _col) const
{
	auto* picker = GetColorPicker(_row, _col);
	if (!picker) return {};
	return picker->getColor2();
}

void CQtTable::SetRowBackgroundColor(const QColor& _color, int _row) const
{
	if (_row >= rowCount()) return;
	for (int i = 0; i < columnCount(); ++i)
		item(_row, i)->setBackgroundColor(_color);
}

void CQtTable::SetColumnBackgroundColor(const QColor& _color, int _col) const
{
	if (_col >= columnCount()) return;
	for (int i = 0; i < rowCount(); ++i)
		item(i, _col)->setBackgroundColor(_color);
}

std::pair<int, int> CQtTable::CurrentCellPos() const
{
	return { currentRow(), currentColumn() };
}

void CQtTable::RestoreSelectedCell(const std::pair<int, int>& _cellPos)
{
	RestoreSelectedCell(_cellPos.first, _cellPos.second);
}

void CQtTable::RestoreSelectedCell(int _row, int _col)
{
	if (_row < rowCount())
		setCurrentCell(_row, _col, QItemSelectionModel::SelectCurrent);
	else
		setCurrentCell(rowCount() - 1, _col, QItemSelectionModel::SelectCurrent);
}

void CQtTable::SetItemUserData(int _row, int _col, const QVariant& _data) const
{
	if (QTableWidgetItem* item = this->item(_row, _col))
		return item->setData(Qt::UserRole, _data);
}

QVariant CQtTable::GetItemUserData(int _row /*= -1*/, int _col /*= -1*/) const
{
	const int row = _row == -1 ? currentRow() : _row;
	const int col = _col == -1 ? currentColumn() : _col;
	if (const QTableWidgetItem* pItem = item(row, col))
		return pItem->data(Qt::UserRole);
	return {};
}

void CQtTable::ShowRow(int _row, bool _show)
{
	if (_show)	showRow(_row);
	else		hideRow(_row);
}

void CQtTable::ShowCol(int _col, bool _show)
{
	if (_show)	showColumn(_col);
	else		hideColumn(_col);
}

void CQtTable::mousePressEvent(QMouseEvent* _event)
{
	if (!indexAt(_event->pos()).isValid())
		emit EmptyAreaClicked();
	QTableWidget::mousePressEvent(_event);
}

void CQtTable::keyPressEvent(QKeyEvent *event)
{
	if (event->key() == Qt::Key_Backspace)
	{
		Clear();
		QTableWidget::keyPressEvent(event);
	}
	else if (event->matches(QKeySequence::Copy))
		Copy();
	else if (event->matches(QKeySequence::Paste))
		Paste();
	else
		QTableWidget::keyPressEvent(event);
}
