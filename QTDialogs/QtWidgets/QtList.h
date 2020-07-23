/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <QListWidget>

class CQtList : public QListWidget
{
	Q_OBJECT

public:
	CQtList(QWidget* _parent);

	void InsertItemCheckable(int _row, const QString& _text, bool _checked = false, const QVariant& _userData = -1);
	void InsertItemCheckable(int _row, const std::string& _text, bool _checked = false, const QVariant& _userData = -1);
	void AddItemCheckable(const QString& _text, bool _checked = false, const QVariant& _userData = -1);
	void SetItemChecked(int _row, bool _checked) const;
	bool GetItemChecked(int _row) const;
};
