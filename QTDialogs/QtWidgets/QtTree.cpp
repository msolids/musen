/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "QtTree.h"

CQtTree::CQtTree(QWidget* _parent) :
	QTreeWidget{ _parent }
{
}

void CQtTree::SetUnitConverter(const CUnitConvertor* _unitConverter)
{
	m_unitConverter = _unitConverter;
}

void CQtTree::SetHeaderText(QTreeWidgetItem* _item, const std::string& _text, EUnitType _units/* = EUnitType::NONE*/) const
{
	_item->setText(0, QString::fromStdString(_text + (m_unitConverter && _units != EUnitType::NONE ? " [" + m_unitConverter->GetSelectedUnit(_units) + "]" : "")));
}

QTreeWidgetItem* CQtTree::CreateItem(int _col, const std::string& _text, EFlags _flags, const QVariant& _userData)
{
	return CreateItem<QTreeWidget>(this, _col, _text, _flags, _userData);
}

QTreeWidgetItem* CQtTree::CreateItem(QTreeWidgetItem* _parent, int _col, const std::string& _text, EFlags _flags, const QVariant& _userData)
{
	return CreateItem<QTreeWidgetItem>(_parent, _col, _text, _flags, _userData);
}

QComboBox* CQtTree::AddComboBox(QTreeWidgetItem* _item, int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, const QVariant& _selected)
{
	auto* combo = new QComboBox{ this };
	int selected = -1;
	for (size_t i = 0; i < _names.size(); ++i)
	{
		combo->addItem(_names[i], i < _data.size() ? _data[i] : QVariant{});
		if (_data[i] == _selected)
			selected = static_cast<int>(i);
	}
	combo->setCurrentIndex(selected);
	setItemWidget(_item, _col, combo);
	return combo;
}

void CQtTree::SetupComboBox(QTreeWidgetItem* _item, int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, const QVariant& _selected) const
{
	auto* combo = dynamic_cast<QComboBox*>(itemWidget(_item, _col));
	if (!combo) return;
	QSignalBlocker blocker{ combo };
	combo->clear();
	int selected = -1;
	for (size_t i = 0; i < _names.size(); ++i)
	{
		combo->addItem(_names[i], i < _data.size() ? _data[i] : QVariant{});
		if (_data[i] == _selected)
			selected = static_cast<int>(i);
	}
	combo->setCurrentIndex(selected);
}

void CQtTree::SetupComboBox(QTreeWidgetItem* _item, int _col, const std::vector<std::string>& _names, const std::vector<std::string>& _data, const std::string& _selected) const
{
	if (_names.size() != _data.size()) return;
	std::vector<QString> names;
	std::vector<QVariant> data;
	for (int i = 0; i < static_cast<int>(_names.size()); ++i)
	{
		names.emplace_back(QString::fromStdString(_names[i]));
		data.emplace_back(QString::fromStdString(_data[i]));
	}
	SetupComboBox(_item, _col, names, data, QString::fromStdString(_selected));
}

void CQtTree::SetComboBoxValue(QTreeWidgetItem* _item, int _col, const QVariant& _value) const
{
	auto* combo = dynamic_cast<QComboBox*>(itemWidget(_item, _col));
	if (!combo) return;
	QSignalBlocker blocker{ combo };
	for (int i = 0; i < combo->count(); ++i)
		if (combo->itemData(i) == _value)
		{
			combo->setCurrentIndex(i);
			return;
		}
	combo->setCurrentIndex(-1);
}

QVariant CQtTree::GetComboBoxValue(QTreeWidgetItem* _item, int _col) const
{
	const auto* combo = dynamic_cast<QComboBox*>(itemWidget(_item, _col));
	if (!combo) return{};
	return combo->itemData(combo->currentIndex());
}

CColorView* CQtTree::AddColorPicker(QTreeWidgetItem* _item, int _col, const CColor& _color)
{
	auto* picker = new CColorView{ this };
	picker->SetColor(_color);
	setItemWidget(_item, _col, picker);
	return picker;
}

void CQtTree::SetColorPickerValue(QTreeWidgetItem* _item, int _col, const CColor& _color) const
{
	auto* picker = dynamic_cast<CColorView*>(itemWidget(_item, _col));
	if (!picker) return;
	QSignalBlocker blocker{ picker };
	picker->SetColor(_color);
}

CColor CQtTree::GetColorPickerValue(QTreeWidgetItem* _item, int _col) const
{
	const auto* picker = dynamic_cast<CColorView*>(itemWidget(_item, _col));
	if (!picker) return {};
	return picker->getColor2();
}

CQtListSpinBox* CQtTree::AddListSpinBox(QTreeWidgetItem* _item, int _col, const std::vector<int>& _values, int _value)
{
	auto* spinbox = new CQtListSpinBox{ _values, this };
	spinbox->SetValue(_value);
	setItemWidget(_item, _col, spinbox);
	return spinbox;
}

void CQtTree::SetupListSpinBox(QTreeWidgetItem* _item, int _col, const std::vector<int>& _values, int _value) const
{
	auto* spinbox = dynamic_cast<CQtListSpinBox*>(itemWidget(_item, _col));
	if (!spinbox) return;
	QSignalBlocker blocker{ spinbox };
	spinbox->SetList(_values);
	spinbox->SetValue(_value);
}

void CQtTree::SetListSpinBoxValue(QTreeWidgetItem* _item, int _col, int _value) const
{
	auto* spinbox = dynamic_cast<CQtListSpinBox*>(itemWidget(_item, _col));
	if (!spinbox) return;
	QSignalBlocker blocker{ spinbox };
	spinbox->SetValue(_value);
}

int CQtTree::GetListSpinBoxValue(QTreeWidgetItem* _item, int _col) const
{
	const auto* spinbox = dynamic_cast<CQtListSpinBox*>(itemWidget(_item, _col));
	if (!spinbox) return {};
	return spinbox->value();
}

CQtDoubleSpinBox* CQtTree::AddDoubleSpinBox(QTreeWidgetItem* _item, int _col, double _value, EUnitType _units/* = EUnitType::NONE*/)
{
	auto* spinbox = new CQtDoubleSpinBox{ m_unitConverter ? m_unitConverter->GetValue(_units, _value) : _value, this };
	setItemWidget(_item, _col, spinbox);
	_item->setData(_col, Qt::UserRole + 1, E2I(_units));
	return spinbox;
}

void CQtTree::SetDoubleSpinBoxValue(QTreeWidgetItem* _item, int _col, double _value) const
{
	auto* spinbox = dynamic_cast<CQtDoubleSpinBox*>(itemWidget(_item, _col));
	if (!spinbox) return;
	const auto units = static_cast<EUnitType>(_item->data(_col, Qt::UserRole + 1).toUInt());
	spinbox->SetValue(m_unitConverter ? m_unitConverter->GetValue(units, _value) : _value);
}

double CQtTree::GetDoubleSpinBoxValue(QTreeWidgetItem* _item, int _col) const
{
	const auto* spinbox = dynamic_cast<CQtDoubleSpinBox*>(itemWidget(_item, _col));
	if (!spinbox) return {};
	const auto units = static_cast<EUnitType>(_item->data(_col, Qt::UserRole + 1).toUInt());
	return m_unitConverter ? m_unitConverter->GetValueSI(units, spinbox->GetValue()) : spinbox->GetValue();
}

QPushButton* CQtTree::AddPushButton(QTreeWidgetItem* _item, int _col, const QString& _text)
{
	auto* button = new QPushButton{ _text, this };
	button->setAutoDefault(false);
	setItemWidget(_item, _col, button);
	return button;
}

QTreeWidgetItem* CQtTree::GetItem(const QVariant& _userData) const
{
	for (auto* item : findItems(QString("*"), Qt::MatchWrap | Qt::MatchWildcard | Qt::MatchRecursive))
		if (item->data(0, Qt::UserRole) == _userData)
			return item;
	return nullptr;
}

void CQtTree::Clear(QTreeWidgetItem* _item)
{
	for (auto* item : _item->takeChildren())
		delete item;
}

QString CQtTree::GetData(QTreeWidgetItem* _item)
{
	if (!_item) return {};
	return _item->data(0, Qt::UserRole).toString();
}

QString CQtTree::GetCurrentData() const
{
	if (!currentItem()) return {};
	return currentItem()->data(0, Qt::UserRole).toString();
}

void CQtTree::SetCurrentItem(const QVariant& _userData)
{
	setCurrentItem(GetItem(_userData));
}

template <typename T>
QTreeWidgetItem* CQtTree::CreateItem(T* _parent, int _col, const std::string& _text, EFlags _flags, const QVariant& _userData)
{
	auto* item = new QTreeWidgetItem{ _parent };
	if (_userData != -1)
		item->setData(_col, Qt::UserRole, _userData);
	item->setText(_col, QString::fromStdString(_text));
	if (_flags != EFlags::Default)
	{
		if (Contains(_flags, EFlags::Edit))		item->setFlags(item->flags() | Qt::ItemIsEditable);
		if (Contains(_flags, EFlags::NoEdit))	item->setFlags(item->flags() & ~Qt::ItemIsEditable);
		if (Contains(_flags, EFlags::Select))	item->setFlags(item->flags() | Qt::ItemIsSelectable);
		if (Contains(_flags, EFlags::NoSelect))	item->setFlags(item->flags() & ~Qt::ItemIsSelectable);
	}
	return item;
}

bool CQtTree::Contains(EFlags _composition, EFlags _flag)
{
	using type = std::underlying_type<EFlags>::type;
	return static_cast<type>(_composition) & static_cast<type>(_flag);
}

CQtTree::EFlags operator|(CQtTree::EFlags _f1, CQtTree::EFlags _f2)
{
	using type = std::underlying_type<CQtTree::EFlags>::type;
	return static_cast<CQtTree::EFlags>(static_cast<type>(_f1) | static_cast<type>(_f2));
}
