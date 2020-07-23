/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "QtList.h"

CQtList::CQtList(QWidget* _parent)
	: QListWidget(_parent)
{
}

void CQtList::InsertItemCheckable(int _row, const QString& _text, bool _checked, const QVariant& _userData)
{
	const bool exist = this->item(_row) != nullptr;
	auto* item = exist ? this->item(_row) : new QListWidgetItem(_text);
	if (exist) item->setText(_text);
	if (_userData != -1) item->setData(Qt::UserRole, _userData);
	item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
	item->setCheckState(_checked ? Qt::Checked : Qt::Unchecked);
	if (!exist)
		insertItem(_row, item);
}

void CQtList::InsertItemCheckable(int _row, const std::string& _text, bool _checked, const QVariant& _userData)
{
	InsertItemCheckable(count(), QString::fromStdString(_text), _checked, _userData);
}

void CQtList::AddItemCheckable(const QString& _text, bool _checked, const QVariant& _userData)
{
	InsertItemCheckable(count(), _text, _checked, _userData);
}

void CQtList::SetItemChecked(int _row, bool _checked) const
{
	if (item(_row))
		item(_row)->setCheckState(_checked ? Qt::Checked : Qt::Unchecked);
}

bool CQtList::GetItemChecked(int _row) const
{
	if (item(_row))
		return item(_row)->checkState() == Qt::Checked;
	return false;
}
