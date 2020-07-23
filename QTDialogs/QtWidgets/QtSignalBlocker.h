/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <QObject>

class CQtSignalBlocker
{
private:
	std::vector<QObject*> m_objects;
	std::vector<bool> m_flags;
public:
	CQtSignalBlocker(QObject* _object);
	CQtSignalBlocker(std::initializer_list<QObject*> _objects);
	~CQtSignalBlocker();

	CQtSignalBlocker(const CQtSignalBlocker& _other) = delete;
	CQtSignalBlocker(CQtSignalBlocker&& _other) noexcept = delete;
	CQtSignalBlocker& operator=(const CQtSignalBlocker& _other) = delete;
	CQtSignalBlocker& operator=(CQtSignalBlocker&& _other) noexcept = delete;
};
