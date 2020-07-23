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
	bool bOldBlock = blockSignals(m_bBlockingPaste);

	QModelIndexList indexes = this->selectionModel()->selection().indexes();

	m_pasteInfo.firstModifiedRow = indexes.count() ? indexes.front().row() : 0;
	m_pasteInfo.lastModifiedRow = 0;

	for (int i = 0; i < indexes.size(); ++i)
	{
		this->item(indexes[i].row(), indexes[i].column())->setText("");

		if (m_bBlockingPaste)
		{
			if (indexes[i].row() < m_pasteInfo.firstModifiedRow)
				m_pasteInfo.firstModifiedRow = indexes[i].row();
			if (indexes[i].row() > m_pasteInfo.lastModifiedRow)
				m_pasteInfo.lastModifiedRow = indexes[i].row();
		}
	}

	blockSignals(bOldBlock);
	emit DataPasted();
}

void CQtTable::Copy()
{
	QModelIndexList indexes = this->selectionModel()->selection().indexes();
	QString str;
	for( int i=indexes.front().row(); i<=indexes.back().row(); ++i )
	{
		for( int j=indexes.front().column(); j<=indexes.back().column(); ++j )
		{
			str += this->item( i, j )->text();
			str += "\t";
		}
		str += "\n";
	}

	QApplication::clipboard()->setText(str);
}

void CQtTable::Paste()
{
	bool bOldBlock = blockSignals( m_bBlockingPaste );
	QString sSelectedText = QApplication::clipboard()->text();
	QStringList rows = sSelectedText.split( QRegExp( QLatin1String("\n") ) );
	while( !rows.empty() && rows.back().size() == 0 )
		rows.pop_back();

	QModelIndexList indexes = this->selectionModel()->selection().indexes();
	int nFirstRow = 0;
	int nFirstColumn = 0;
	if ( indexes.count() )
	{
		nFirstRow = indexes.at(0).row();
		nFirstColumn = indexes.at(0).column();
	}

	if ( nFirstRow < 0 ) nFirstRow = 0;
	int nRowMax = rows.count();
	if( nRowMax > this->rowCount() - nFirstRow )
		nRowMax = this->rowCount() - nFirstRow;
	for( int i=0; i<nRowMax; i++ )
	{
		QStringList columns = rows[i].split(QRegExp("[\t ]"));
		while( !columns.empty() && columns.back().size() == 0 )
			columns.pop_back();
		int nColumnMax = columns.count();
		if( nColumnMax > this->columnCount() - nFirstColumn )
			nColumnMax = this->columnCount() - nFirstColumn;
		for( int j=0; j<nColumnMax; ++j )
			this->item( i+nFirstRow, j+nFirstColumn )->setText( QString::number( columns[j].toDouble() ) );
	}
	m_pasteInfo.firstModifiedRow = nFirstRow;
	m_pasteInfo.lastModifiedRow  = nFirstRow + nRowMax;
	blockSignals( bOldBlock );
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
	delete item(_row, _col);
	auto *widget = new QWidget(this);
	auto *checkBox = new QCheckBox(widget);
	auto *layout = new QHBoxLayout(widget);
	layout->addWidget(checkBox);
	layout->setAlignment(Qt::AlignCenter);
	layout->setContentsMargins(0, 0, 0, 0);
	widget->setLayout(layout);
	checkBox->setChecked(_checked);
	checkBox->setObjectName("CheckBox");
	connect(checkBox, &QCheckBox::stateChanged, this, [=] { CheckBoxStateChanged(_row, _col, checkBox); });
	connect(checkBox, &QCheckBox::stateChanged, this, [=] { setCurrentCell(_row, _col); });
	setCellWidget(_row, _col, widget);
	return checkBox;
}

QCheckBox* CQtTable::GetCheckBox(int _row, int _col) const
{
	return cellWidget(_row, _col)->findChild<QCheckBox*>("CheckBox");
}

void CQtTable::SetCheckBoxChecked(int _row, int _col, bool _checked) const
{
	auto *checkBox = cellWidget(_row, _col)->findChild<QCheckBox*>("CheckBox");
	if (!checkBox) return;
	QSignalBlocker blocker{ checkBox };
	checkBox->setChecked(_checked);
}

bool CQtTable::GetCheckBoxChecked(int _row, int _col) const
{
	auto *checkBox = cellWidget(_row, _col)->findChild<QCheckBox*>("CheckBox");
	if (!checkBox) return false;
	return checkBox->isChecked();
}

QComboBox* CQtTable::SetComboBox(const int _row, const int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, int _iSelected)
{
	delete item(_row, _col);
	auto* comboBox = new QComboBox(this);
	for (size_t i = 0; i < _names.size(); ++i)
		comboBox->insertItem(comboBox->count(), _names[i], i < _data.size() ? _data[i] : QVariant{});
	comboBox->setCurrentIndex(_iSelected);
	connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [=] { ComboBoxIndexChanged(_row, _col, comboBox); });
	connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [=] { setCurrentCell(_row, _col); });
	setCellWidget(_row, _col, comboBox);
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

QVariant CQtTable::GetComboBoxValue(int _row, int _col) const
{
	auto* combo = dynamic_cast<QComboBox*>(cellWidget(_row, _col));
	if (!combo) return{};
	return combo->itemData(combo->currentIndex());
}

QProgressBar* CQtTable::SetProgressBar(int _row, int _col, int _value)
{
	delete item(_row, _col);
	auto* bar = new QProgressBar(this);
	bar->setRange(0, 100);
	bar->setValue(_value);
	setCellWidget(_row, _col, bar);
	return bar;
}

QProgressBar* CQtTable::GetProgressBar(int _row, int _col) const
{
	return dynamic_cast<QProgressBar*>(cellWidget(_row, _col));
}

void CQtTable::SetProgressBarValue(int _row, int _col, int _value) const
{
	auto* bar = dynamic_cast<QProgressBar*>(cellWidget(_row, _col));
	if (!bar) return;
	bar->setValue(_value);
}

void CQtTable::SetRowBackgroundColor(const QColor& _color, int _row) const
{
	if (_row >= rowCount()) return;
	for (int i = 0; i < columnCount(); ++i)
		item(_row, i)->setBackgroundColor(_color);
}

void CQtTable::SetColumnBackgroundColor(const QColor& _color, int _nColumn) const
{
	if (_nColumn >= columnCount()) return;
	for (int i = 0; i < rowCount(); ++i)
		item(i, _nColumn)->setBackgroundColor(_color);
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

bool CQtTable::blockSignals(bool _flag)
{
	QList<QCheckBox*> listCheckBox = findChildren<QCheckBox*>(QString(), Qt::FindChildrenRecursively);
	for (auto& cb : listCheckBox)
		cb->blockSignals(_flag);
	QList<QComboBox*> listComboBox = findChildren<QComboBox*>(QString(), Qt::FindChildrenRecursively);
	for (auto& cb : listComboBox)
		cb->blockSignals(_flag);
	return QTableWidget::blockSignals(_flag);
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
