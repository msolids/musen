/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QTreeWidget>
#include <QComboBox>
#include <QPushButton>
#include "UnitConvertor.h"
#include "ColorView.h"
#include "QtListSpinBox.h"
#include "QtDoubleSpinBox.h"

class CQtTree : public QTreeWidget
{
	const CUnitConvertor* m_unitConverter{};

public:
	enum class EFlags
	{
		Default   = 1 << 0,
		Edit      = 1 << 1,
		NoEdit    = 1 << 2,
		Select    = 1 << 3,
		NoSelect  = 1 << 4,
	};

	CQtTree(QWidget* _parent = nullptr);

	// Sets unit converter.
	void SetUnitConverter(const CUnitConvertor* _unitConverter);

	// Sets text of the header column of the given item.
	void SetHeaderText(QTreeWidgetItem* _item, const std::string& _text, EUnitType _units = EUnitType::NONE) const;

	// Creates a new item as a child of the tree.
	QTreeWidgetItem* CreateItem(int _col, const std::string& _text, EFlags _flags = EFlags::Default, const QVariant& _userData = -1);
	// Creates a new item as a child of the widget.
	QTreeWidgetItem* CreateItem(QTreeWidgetItem* _parent, int _col, const std::string& _text, EFlags _flags = EFlags::Default, const QVariant& _userData = -1);

	// Adds a combo box to the selected column of the given existing item.
	QComboBox* AddComboBox(QTreeWidgetItem* _item, int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, const QVariant& _selected);
	// Configures a combo box widget at the selected column of the given existing item.
	void SetupComboBox(QTreeWidgetItem* _item, int _col, const std::vector<QString>& _names, const std::vector<QVariant>& _data, const QVariant& _selected) const;
	// Configures a combo box widget at the selected column of the given existing item.
	void SetupComboBox(QTreeWidgetItem* _item, int _col, const std::vector<std::string>& _names, const std::vector<std::string>& _data, const std::string& _selected) const;
	// Sets a value of the combo box widget at the selected column of the given existing item.
	void SetComboBoxValue(QTreeWidgetItem* _item, int _col, const QVariant& _value) const;
	// Returns a value of the combo box widget at the selected column of the given existing item.
	QVariant GetComboBoxValue(QTreeWidgetItem* _item, int _col) const;

	// Adds a color picker widget to the selected column of the given existing item.
	CColorView* AddColorPicker(QTreeWidgetItem* _item, int _col, const CColor& _color);
	// Sets a value of the color picker widget at the selected column of the given existing item.
	void SetColorPickerValue(QTreeWidgetItem* _item, int _col, const CColor& _color) const;
	// Returns a value of the color picker widget at the selected column of the given existing item.
	CColor GetColorPickerValue(QTreeWidgetItem* _item, int _col) const;

	// Adds a list spin box widget to the selected column of the given existing item.
	CQtListSpinBox* AddListSpinBox(QTreeWidgetItem* _item, int _col, const std::vector<int>& _values, int _value);
	// Configures a list spin box widget at the selected column of the given existing item.
	void SetupListSpinBox(QTreeWidgetItem* _item, int _col, const std::vector<int>& _values, int _value) const;
	// Sets a value of the list spin box widget at the selected column of the given existing item.
	void SetListSpinBoxValue(QTreeWidgetItem* _item, int _col, int _value) const;
	// Returns a value of the list spin box widget at the selected column of the given existing item.
	int GetListSpinBoxValue(QTreeWidgetItem* _item, int _col) const;

	// Adds a double spin box widget to the selected column of the given existing item.
	CQtDoubleSpinBox* AddDoubleSpinBox(QTreeWidgetItem* _item, int _col, double _value, EUnitType _units = EUnitType::NONE);
	// Sets a value of the double spin box widget at the selected column of the given existing item.
	void SetDoubleSpinBoxValue(QTreeWidgetItem* _item, int _col, double _value) const;
	// Returns a (unit-converted) value of the double spin box widget at the selected column of the given existing item.
	double GetDoubleSpinBoxValue(QTreeWidgetItem* _item, int _col) const;

	// Adds a push button widget to the selected column of the given existing item.
	QPushButton* AddPushButton(QTreeWidgetItem* _item, int _col, const QString& _text);

	// Returns an item with the specified user data.
	QTreeWidgetItem* GetItem(const QVariant& _userData) const;

	// Removes all child items of the specified item.
	static void Clear(QTreeWidgetItem* _item);

	// Returns user data of the specified item.
	static QString GetData(QTreeWidgetItem* _item);
	// Returns user data of the current item.
	QString GetCurrentData() const;
	// Sets an item with the specified user data as current.
	void SetCurrentItem(const QVariant& _userData);

private:
	// Creates a new item as a child of the widget.
	template<typename T>
	QTreeWidgetItem* CreateItem(T* _parent, int _col, const std::string& _text, EFlags _flags = EFlags::Default, const QVariant& _userData = -1);

	// Checks whether the composition of flags contains a flag.
	static bool Contains(EFlags _composition, EFlags _flag);
};

CQtTree::EFlags operator|(CQtTree::EFlags _f1, CQtTree::EFlags _f2);
